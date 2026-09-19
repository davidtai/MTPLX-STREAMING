#!/usr/bin/env python3
"""F13 reader-burst microbenchmark for the DSV4.1 Q4 streaming-decode reader.

Faithful user-space reproduction (no MLX, no model) of the retained expert
reader's per-layer read burst, to test whether the ~0.3 ms of main-thread
Python/graph-building after read submission (which holds the CPython GIL)
delays the reader threads from reaching os.preadv -- idling the SSD at the
start of every burst -- and to measure what recovers it.

Structure reproduced (docs/.../packed/plane_lane.py::bind_reader and
mtplx/expert_io.py::_readv_range_into):
  * A layer burst demand-reads ~4 expert records = 12 "planes" of 5,898,240 B.
  * Records are split across part threads at 3 records/part -> parts of 3 + 1.
  * Each part thread submits all-but-one of its planes to a shared pool
    (15 workers at the recommended fanout=4 arm) and reads one plane INLINE.
  * Each plane is ONE os.preadv of 5,898,240 B from a Python thread.
  * Right after submitting, the MAIN thread does ~0.30 ms of pure-Python work
    holding the GIL, then blocks on the parts.

Variants (see build_variants): A baseline (A0 block-now / A1 spin-then-block),
B sys.setswitchinterval sweep, C sleep(0)/sched_yield nudges, D plane-split
factors, E a native (ctypes) pthread reader, F steady-state ceiling.

Reported per variant: mean/p50/p90 burst ms, effective GB/s, and (Python
variants) the submit->first-preadv delay in us (min preadv-entry stamp minus
submit time) -- the direct GIL-hypothesis probe. Native reports the same delay
measured entirely in C's CLOCK_MONOTONIC.

Operational: read-only on experts.bin with F_NOCACHE + read-ahead off, into
page-aligned anonymous mmap buffers. Before every variant the GPU exclusive
lock is checked (lsof); a held lock is polled every 60 s up to 60 min and the
variant runs only while free. Each variant is bounded to <= ~22 GB and a few
seconds. Everything is meant to run under `nice -n 19`; CPU only.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import fcntl
import json
import mmap
import os
import random
import resource
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

# --- fixed geometry of the real artifact -----------------------------------
EXPERTS_PATH = os.path.expanduser(
    "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-mxfp4/experts.bin"
)
RECORD_STRIDE = 18_800_640          # bytes between successive expert records
PLANE_OFFSETS = (0, 6_266_880, 12_533_760)  # gate / up / down within a record
PLANE_SIZE = 5_898_240              # bytes per plane (one os.preadv)
RECORDS_PER_BURST = 4               # ~4.08 records demanded per layer call
PART_RECORDS = (3, 1)               # 3 records/part -> parts of 3 + 1
POOL_BASE = 15                      # fanout pool width at the fanout=4 arm

F_NOCACHE = getattr(fcntl, "F_NOCACHE", 48)
F_RDAHEAD = 45                      # macOS F_RDAHEAD (not exposed by fcntl)

LOCK_PATH = "/tmp/mtplx-gpu-exclusive.lock"
FLAG_PATH = "/tmp/dsv41-fable-window.active"   # Fable's active-window flag
LOCK_POLL_S = 60
LOCK_MAX_WAIT_S = 90 * 60                       # poll up to 90 min per variant

DEFAULT_SWITCH_INTERVAL = 5e-3
SPIN_NS = 300_000                   # ~0.30 ms main-thread hold after submit
GAP_S = 0.0025                      # ~2.5 ms idle between bursts (GPU phase)
WARM = 10
MEAS = 300                          # >=300 bursts; 310 total ~= 21.9 GB/variant


# --- GPU-window gate --------------------------------------------------------
# Gate (coordinator, 2026-09-19): run whenever Fable's active-window flag file
# is ABSENT. The GPU lock is intentionally ignored -- another session's trainer
# holds it almost continuously but is GPU-bound / not SSD-sensitive, and Fable's
# own windows always set the flag. Checked immediately before every variant;
# the series stops as soon as the flag appears.
def lock_held() -> bool:
    if not os.path.exists(LOCK_PATH):
        return False
    try:
        r = subprocess.run(["lsof", LOCK_PATH], capture_output=True, text=True,
                            timeout=30)
    except Exception:
        return True
    return r.returncode == 0 and bool(r.stdout.strip())


def window_blocked() -> str:
    if os.path.exists(FLAG_PATH):
        return "fable-window-flag"
    return ""


def wait_for_window_free(label: str) -> bool:
    reason = window_blocked()
    if not reason:
        return True
    waited = 0
    print(f"[gate] blocked ({reason}); waiting to run {label} "
          f"(poll {LOCK_POLL_S}s, max {LOCK_MAX_WAIT_S // 60}min)...", flush=True)
    while True:
        reason = window_blocked()
        if not reason:
            break
        if waited >= LOCK_MAX_WAIT_S:
            print(f"[gate] still blocked ({reason}) after {waited}s; "
                  f"skipping {label}", flush=True)
            return False
        time.sleep(LOCK_POLL_S)
        waited += LOCK_POLL_S
    print(f"[gate] clear after {waited}s; running {label}", flush=True)
    return True


# --- helpers ----------------------------------------------------------------
def busy_spin(target_ns: int) -> int:
    """Pure-Python arithmetic loop holding the GIL for ~target_ns."""
    t0 = time.perf_counter_ns()
    acc = 0
    while time.perf_counter_ns() - t0 < target_ns:
        for _ in range(200):
            acc = (acc * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    return acc


def aligned_ranges(total: int, parts: int, align: int = 4096):
    """Tile [0, total) into up to `parts` contiguous (start, length) ranges with
    interior cuts rounded to `align` (matches expert_io._fanout_byte_ranges)."""
    if parts <= 1 or total <= 0:
        return [(0, total)]
    out = []
    start = 0
    for i in range(1, parts):
        cut = (total * i) // parts
        cut = (cut // align) * align
        if cut <= start or cut >= total:
            continue
        out.append((start, cut - start))
        start = cut
    out.append((start, total - start))
    return out


def stats(ms_list, first_us_list, bytes_per_burst):
    a = np.asarray(ms_list, dtype=np.float64)
    out = {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "min_ms": float(a.min()),
        "gb_s": float(bytes_per_burst / (a.mean() / 1e3) / 1e9),
    }
    if first_us_list is not None and len(first_us_list):
        f = np.asarray(first_us_list, dtype=np.float64)
        out.update({
            "first_preadv_us_mean": float(f.mean()),
            "first_preadv_us_p50": float(np.percentile(f, 50)),
            "first_preadv_us_p90": float(np.percentile(f, 90)),
        })
    return out


# --- benchmark configuration & shared buffers -------------------------------
@dataclass
class Config:
    path: str = EXPERTS_PATH
    stride: int = RECORD_STRIDE
    plane_offsets: tuple = PLANE_OFFSETS
    plane_size: int = PLANE_SIZE
    warm: int = WARM
    meas: int = MEAS
    gap_s: float = GAP_S
    spin_ns: int = SPIN_NS
    use_lock_gate: bool = True
    steady_budget_bytes: int = 15_000_000_000   # <= 25 GB/variant cap
    steady_max_wall_s: float = 8.0


@dataclass
class Harness:
    cfg: Config
    fd: int = -1
    n_records: int = 0
    buffers: list = field(default_factory=list)      # 12 mmap plane buffers
    np_views: list = field(default_factory=list)      # numpy uint8 views
    base_ptrs: list = field(default_factory=list)     # ctypes pointers per plane
    dylib: object = None

    def open(self):
        c = self.cfg
        self.fd = os.open(c.path, os.O_RDONLY)
        # F_NOCACHE (bypass unified buffer cache) + read-ahead off.
        fcntl.fcntl(self.fd, F_NOCACHE, 1)
        try:
            fcntl.fcntl(self.fd, F_RDAHEAD, 0)
        except OSError:
            pass
        self.n_records = os.fstat(self.fd).st_size // c.stride
        n_planes = RECORDS_PER_BURST * len(c.plane_offsets)  # 12
        for _ in range(n_planes):
            b = mmap.mmap(-1, c.plane_size)     # page-aligned anonymous buffer
            b[:] = b"\x00" * c.plane_size       # fault in every page up front
            v = np.frombuffer(b, dtype=np.uint8)
            self.buffers.append(b)
            self.np_views.append(v)
            self.base_ptrs.append(v.ctypes.data)

    def load_dylib(self):
        path = str(Path(__file__).with_name("libreader.dylib"))
        lib = ctypes.CDLL(path)  # CDLL releases the GIL around every call
        lib.reader_init.argtypes = [ctypes.c_int]
        lib.reader_init.restype = ctypes.c_int
        lib.submit_batch.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.submit_batch.restype = None
        lib.wait_all.argtypes = []
        lib.wait_all.restype = None
        lib.get_first_pread_delay_ns.argtypes = []
        lib.get_first_pread_delay_ns.restype = ctypes.c_longlong
        lib.get_job_err.argtypes = [ctypes.c_int]
        lib.get_job_err.restype = ctypes.c_int
        lib.reader_shutdown.argtypes = []
        lib.reader_shutdown.restype = None
        self.dylib = lib

    def random_bases(self):
        recs = random.sample(range(self.n_records), RECORDS_PER_BURST)
        return [r * self.cfg.stride for r in recs]

    def close(self):
        for b in self.buffers:
            try:
                b.close()
            except Exception:
                pass
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class Job(ctypes.Structure):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("offset", ctypes.c_longlong),
        ("length", ctypes.c_longlong),
        ("dest", ctypes.c_void_p),
        ("err", ctypes.c_int),
    ]


# --- plane/sub-range job templates ------------------------------------------
def build_templates(cfg: Config, split: int):
    """Return (part0, part1, njobs). Each entry is a per-IO template:
    (const_off, rec_slot, buf_off, length, plane_idx). const_off is the file
    offset within a record (plane offset + sub-range start); the per-burst file
    offset is record_base[rec_slot] + const_off."""
    sub = aligned_ranges(cfg.plane_size, split)
    part = [[], []]
    idx = 0
    for plane in range(RECORDS_PER_BURST * len(cfg.plane_offsets)):  # 0..11
        rec_slot = plane // len(cfg.plane_offsets)
        plane_off = cfg.plane_offsets[plane % len(cfg.plane_offsets)]
        which = 0 if rec_slot < PART_RECORDS[0] else 1
        for (s_start, s_len) in sub:
            part[which].append((plane_off + s_start, rec_slot, plane, s_start,
                                s_len, idx))
            idx += 1
    return part[0], part[1], idx


# --- Python burst variant ---------------------------------------------------
def run_python_variant(h: Harness, name, *, spin_ns, switch_interval,
                       yield_mode=None, yield_count=0, split=1):
    cfg = h.cfg
    p0_t, p1_t, njobs = build_templates(cfg, split)
    pool_size = min(64, max(POOL_BASE, njobs))
    bytes_per_burst = RECORDS_PER_BURST * len(cfg.plane_offsets) * cfg.plane_size

    # Fixed per-split destination memoryviews (only file offset varies/burst).
    mv_by_idx = {}
    for tmpl in (p0_t, p1_t):
        for (_coff, _slot, plane, s_start, s_len, jidx) in tmpl:
            mv_by_idx[jidx] = memoryview(h.buffers[plane])[s_start:s_start + s_len]

    start_ns = [0] * njobs
    end_ns = [0] * njobs
    cur = [None, None]            # current burst jobs per part: [(off, mv, jidx)]
    go = [threading.Event(), threading.Event()]
    done = [threading.Event(), threading.Event()]
    stop = threading.Event()
    fd = h.fd

    pool = ThreadPoolExecutor(max_workers=pool_size,
                              thread_name_prefix="f13-plane")

    def read_job(job):
        off, mv, jidx = job
        start_ns[jidx] = time.perf_counter_ns()
        total = 0
        L = len(mv)
        while total < L:
            r = os.preadv(fd, [mv[total:]], off + total)
            if r <= 0:
                raise IOError(f"short read at {off + total}")
            total += r
        end_ns[jidx] = time.perf_counter_ns()

    def part_worker(pid):
        while True:
            go[pid].wait()
            go[pid].clear()
            if stop.is_set():
                return
            jobs = cur[pid]
            futs = [pool.submit(read_job, j) for j in jobs[1:]]
            read_job(jobs[0])           # one plane read inline on the part thread
            for f in futs:
                f.result()
            done[pid].set()

    threads = [threading.Thread(target=part_worker, args=(i,), daemon=True)
               for i in range(2)]
    for t in threads:
        t.start()

    if yield_mode == "sleep0":
        nudge = lambda: time.sleep(0)
    elif yield_mode == "yield":
        nudge = os.sched_yield
    else:
        nudge = None

    old_si = sys.getswitchinterval()
    sys.setswitchinterval(switch_interval)
    ms_list, first_list = [], []
    try:
        total = cfg.warm + cfg.meas
        for b in range(total):
            bases = h.random_bases()
            for pid, tmpl in ((0, p0_t), (1, p1_t)):
                cur[pid] = [(bases[slot] + coff, mv_by_idx[jidx], jidx)
                            for (coff, slot, _pl, _ss, _sl, jidx) in tmpl]
            submit_t = time.perf_counter_ns()
            go[0].set()
            go[1].set()
            if nudge is not None:
                for _ in range(yield_count):
                    nudge()
            if spin_ns > 0:
                busy_spin(spin_ns)
            done[0].wait()
            done[1].wait()
            done[0].clear()
            done[1].clear()
            if b >= cfg.warm:
                b_end = max(end_ns[:njobs])
                b_first = min(start_ns[:njobs])
                ms_list.append((b_end - submit_t) / 1e6)
                first_list.append((b_first - submit_t) / 1e3)
            time.sleep(cfg.gap_s)
    finally:
        stop.set()
        go[0].set()
        go[1].set()
        for t in threads:
            t.join(timeout=5)
        pool.shutdown(wait=True)
        sys.setswitchinterval(old_si)

    r = stats(ms_list, first_list, bytes_per_burst)
    r.update(kind="python", name=name, split=split, njobs=njobs,
             pool_size=pool_size, switch_interval=switch_interval,
             spin_ns=spin_ns, yield_mode=yield_mode, yield_count=yield_count,
             bytes_per_burst=bytes_per_burst)
    return r


# --- native (ctypes pthread) burst variant ----------------------------------
def run_native_variant(h: Harness, name, *, nthreads, split, spin_ns):
    cfg = h.cfg
    lib = h.dylib
    p0_t, p1_t, njobs = build_templates(cfg, split)
    bytes_per_burst = RECORDS_PER_BURST * len(cfg.plane_offsets) * cfg.plane_size

    jobs_arr = (Job * njobs)()
    consts = [0] * njobs             # const file offset within a record, per job
    slots = [0] * njobs
    for tmpl in (p0_t, p1_t):
        for (coff, slot, plane, s_start, s_len, jidx) in tmpl:
            jobs_arr[jidx].fd = fd_i = h.fd
            jobs_arr[jidx].length = s_len
            jobs_arr[jidx].dest = h.base_ptrs[plane] + s_start
            jobs_arr[jidx].err = 0
            consts[jidx] = coff
            slots[jidx] = slot
    arr_ptr = ctypes.cast(jobs_arr, ctypes.c_void_p)

    rc = lib.reader_init(nthreads)
    if rc != 0:
        raise RuntimeError(f"reader_init({nthreads}) failed rc={rc}")

    ms_list, first_list = [], []
    max_err = 0
    try:
        total = cfg.warm + cfg.meas
        for b in range(total):
            bases = h.random_bases()
            for jidx in range(njobs):
                jobs_arr[jidx].offset = bases[slots[jidx]] + consts[jidx]
            t0 = time.perf_counter_ns()
            lib.submit_batch(arr_ptr, njobs)
            if spin_ns > 0:
                busy_spin(spin_ns)     # A1 main-thread behavior; overlaps I/O
            lib.wait_all()
            t1 = time.perf_counter_ns()
            if b >= cfg.warm:
                ms_list.append((t1 - t0) / 1e6)
                d = lib.get_first_pread_delay_ns()
                if d >= 0:
                    first_list.append(d / 1e3)
                for jidx in range(njobs):
                    e = lib.get_job_err(jidx)
                    if e:
                        max_err = max(max_err, e)
            time.sleep(cfg.gap_s)
    finally:
        lib.reader_shutdown()

    r = stats(ms_list, first_list, bytes_per_burst)
    r.update(kind="native", name=name, split=split, njobs=njobs,
             nthreads=nthreads, spin_ns=spin_ns, max_err=max_err,
             bytes_per_burst=bytes_per_burst)
    return r


# --- steady-state ceiling (variant F) ---------------------------------------
def run_steady(h: Harness, name, *, nthreads, budget_bytes=None,
               max_wall_s=None):
    cfg = h.cfg
    fd = h.fd
    if budget_bytes is None:
        budget_bytes = cfg.steady_budget_bytes
    if max_wall_s is None:
        max_wall_s = cfg.steady_max_wall_s
    n_off = h.n_records
    blocks_per_thread = max(1, int(budget_bytes / cfg.plane_size / nthreads))
    counts = [0] * nthreads
    start_evt = threading.Event()
    deadline = [0.0]

    def worker(tid):
        buf = h.buffers[tid % len(h.buffers)]
        mv = memoryview(buf)
        L = cfg.plane_size
        pmax = len(cfg.plane_offsets)
        start_evt.wait()
        c = 0
        while c < blocks_per_thread and time.perf_counter() < deadline[0]:
            rec = random.randrange(n_off)
            off = rec * cfg.stride + cfg.plane_offsets[random.randrange(pmax)]
            total = 0
            while total < L:
                rr = os.preadv(fd, [mv[total:]], off + total)
                if rr <= 0:
                    raise IOError("short read (steady)")
                total += rr
            c += 1
        counts[tid] = c

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(nthreads)]
    for t in threads:
        t.start()
    t0 = time.perf_counter()
    deadline[0] = t0 + max_wall_s
    start_evt.set()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    total_bytes = sum(counts) * cfg.plane_size
    return {
        "kind": "steady", "name": name, "nthreads": nthreads,
        "n": sum(counts), "wall_s": wall,
        "gb_s": float(total_bytes / wall / 1e9),
        "total_gb": float(total_bytes / 1e9),
        "block_bytes": cfg.plane_size,
    }


# --- variant table ----------------------------------------------------------
def _si_name(si):
    return f"B_switch_{si:.0e}".replace("e-0", "e-")


def build_variants(subset=None):
    DS = DEFAULT_SWITCH_INTERVAL
    # Priority order (coordinator): A0, A1, B, E native N=12 s1/s2, F, then rest.
    # A run may be cut short when Fable's flag reappears, so the highest-value
    # variants come first.
    v = [
        # --- priority head ---
        ("python", "A0_block_now", dict(spin_ns=0, switch_interval=DS)),
        ("python", "A1_spin_0p30ms", dict(spin_ns=SPIN_NS, switch_interval=DS)),
        ("python", _si_name(5e-4), dict(spin_ns=SPIN_NS, switch_interval=5e-4)),
        ("python", _si_name(5e-5), dict(spin_ns=SPIN_NS, switch_interval=5e-5)),
        ("python", _si_name(1e-5), dict(spin_ns=SPIN_NS, switch_interval=1e-5)),
        ("native", "E_native_n12_s1", dict(nthreads=12, split=1, spin_ns=SPIN_NS)),
        ("native", "E_native_n12_s2", dict(nthreads=12, split=2, spin_ns=SPIN_NS)),
        ("steady", "F_steady_8", dict(nthreads=8)),
        ("steady", "F_steady_12", dict(nthreads=12)),
        # --- the rest ---
        ("python", "C_yield_x1",
         dict(spin_ns=SPIN_NS, switch_interval=DS, yield_mode="yield", yield_count=1)),
        ("python", "C_sleep0_x1",
         dict(spin_ns=SPIN_NS, switch_interval=DS, yield_mode="sleep0", yield_count=1)),
        ("python", "C_yield_x8",
         dict(spin_ns=SPIN_NS, switch_interval=DS, yield_mode="yield", yield_count=8)),
        ("python", "D_split2", dict(spin_ns=SPIN_NS, switch_interval=DS, split=2)),
        ("python", "D_split4", dict(spin_ns=SPIN_NS, switch_interval=DS, split=4)),
        ("native", "E_native_n12_s4", dict(nthreads=12, split=4, spin_ns=SPIN_NS)),
        ("native", "E_native_n8_s1", dict(nthreads=8, split=1, spin_ns=SPIN_NS)),
        ("native", "E_native_n8_s2", dict(nthreads=8, split=2, spin_ns=SPIN_NS)),
        ("native", "E_native_n8_s4", dict(nthreads=8, split=4, spin_ns=SPIN_NS)),
        ("native", "E_native_n16_s1", dict(nthreads=16, split=1, spin_ns=SPIN_NS)),
        ("native", "E_native_n16_s2", dict(nthreads=16, split=2, spin_ns=SPIN_NS)),
        ("native", "E_native_n16_s4", dict(nthreads=16, split=4, spin_ns=SPIN_NS)),
    ]
    if subset:
        wanted = set(subset)
        v = [x for x in v if x[1] in wanted]
    return v


def dispatch(h, kind, name, params):
    if kind == "python":
        return run_python_variant(h, name, **params)
    if kind == "native":
        return run_native_variant(h, name, **params)
    if kind == "steady":
        return run_steady(h, name, **params)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "selftest"], default="real")
    ap.add_argument("--out", default=str(Path(__file__).with_name("results")
                                         / "reader_burst_results.json"))
    ap.add_argument("--variants", default="", help="comma list, else all")
    args = ap.parse_args()

    subset = [s for s in args.variants.split(",") if s] or None

    if args.mode == "selftest":
        # Tiny geometry on a scratch file: exercises every code path with
        # negligible I/O (<50 MB total) and no dependence on experts.bin or the
        # GPU window. Runs while Fable's window flag is set (correctness only).
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "f13_selftest.bin"
        stride = 245_760
        psize = 65_536
        offs = (0, 81_920, 163_840)
        nrec = 64
        with open(tmp, "wb") as fo:
            fo.truncate(nrec * stride)
        cfg = Config(path=str(tmp), stride=stride, plane_offsets=offs,
                     plane_size=psize, warm=2, meas=5, gap_s=0.0005,
                     spin_ns=SPIN_NS, use_lock_gate=False,
                     steady_budget_bytes=3_000_000, steady_max_wall_s=1.0)
        if subset is None:
            # one representative of every code path (~41 MB of reads total)
            subset = ["A0_block_now", "A1_spin_0p30ms", "B_switch_5e-5",
                      "C_yield_x8", "D_split4", "E_native_n8_s1",
                      "E_native_n12_s4", "F_steady_8"]
    else:
        cfg = Config()

    h = Harness(cfg)
    h.open()
    h.load_dylib()
    print(f"[setup] fd open, n_records={h.n_records}, planes={len(h.buffers)}, "
          f"sizeof(Job)={ctypes.sizeof(Job)}, mode={args.mode}", flush=True)

    variants = build_variants(subset)
    results = {
        "meta": {
            "host": os.uname().nodename,
            "mode": args.mode,
            "path": cfg.path,
            "plane_size": cfg.plane_size,
            "record_stride": cfg.stride,
            "plane_offsets": list(cfg.plane_offsets),
            "records_per_burst": RECORDS_PER_BURST,
            "bytes_per_burst": RECORDS_PER_BURST * len(cfg.plane_offsets) * cfg.plane_size,
            "warm": cfg.warm, "meas": cfg.meas, "gap_s": cfg.gap_s,
            "spin_ns": cfg.spin_ns,
            "default_switch_interval": DEFAULT_SWITCH_INTERVAL,
            "n_records": h.n_records,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "variants": [],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        with open(out_path, "w") as fo:
            json.dump(results, fo, indent=2)

    flush()
    try:
        for kind, name, params in variants:
            if cfg.use_lock_gate and not wait_for_window_free(name):
                results["variants"].append({"kind": kind, "name": name,
                                            "skipped": window_blocked() or "blocked",
                                            "utc": utcnow()})
                flush()
                continue
            utc_start = utcnow()
            t0 = time.perf_counter()
            r = dispatch(h, kind, name, params)
            r["run_wall_s"] = time.perf_counter() - t0
            r["utc_start"] = utc_start
            r["utc_end"] = utcnow()
            # residual race guard: a window flag may have appeared during the
            # (<=20 s) run; flag it so the report can mark it for cross-check.
            r["window_flag_after"] = os.path.exists(FLAG_PATH)
            results["variants"].append(r)
            flush()
            if r["kind"] == "steady":
                print(f"[done] {name:22s} {r['gb_s']:6.2f} GB/s steady "
                      f"({r['n']} blocks, {r['wall_s']:.2f}s)", flush=True)
            else:
                fp = r.get("first_preadv_us_mean", float("nan"))
                print(f"[done] {name:22s} mean={r['mean_ms']:.3f}ms "
                      f"p50={r['p50_ms']:.3f} p90={r['p90_ms']:.3f} "
                      f"{r['gb_s']:.2f}GB/s first-preadv={fp:.1f}us "
                      f"(wall {r['run_wall_s']:.1f}s)", flush=True)
    finally:
        results["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        # macOS ru_maxrss is in bytes; confirms the <=300 MB peak-RSS budget.
        results["meta"]["peak_rss_mb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 1)
        flush()
        h.close()
    print(f"[out] {out_path}  peak_rss_mb={results['meta']['peak_rss_mb']}",
          flush=True)


if __name__ == "__main__":
    main()
