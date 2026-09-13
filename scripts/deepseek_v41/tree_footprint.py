#!/usr/bin/env python3
"""Sum phys_footprint over a process tree (W121 gpu_window guard reader).

phys_footprint (mach ``proc_pid_rusage`` RUSAGE_INFO_V4 ``ri_phys_footprint``) is
the kernel's real per-process memory total: it INCLUDES IOAccelerator/Metal (GPU)
pages -- wired OR not -- and the process's dirty anonymous pages, but NOT the
shared file page cache. The guard adds this process accounting value to its
baseline as a conservative estimate. It separately measures live physical used
memory, including file cache, and compares both against its ceiling. A process
tree footprint is not a measurement of system physical used memory.

Verified on the idle box (2026-09-12): Qwen pid phys_footprint 79.73 GiB from this
API == "79 GB IOAccelerator" from ``footprint -p``.  No sudo, cross-process (same
user), CPU-light -- a few syscalls.

Usage:
  tree_footprint.py <root_pid> [<root_pid> ...]   # sum footprint over the tree(s)
  tree_footprint.py --pid <pid>                   # single pid footprint (bytes)
  tree_footprint.py --self-check <pid>            # print "<bytes> footprint_p=<bytes|NA>"
Prints the total in BYTES to stdout, or nothing (exit 2) on a hard failure.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import subprocess
import sys

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_RUSAGE_INFO_V4 = 4


class _RUsageInfoV4(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_ubyte * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


def phys_footprint(pid: int) -> int | None:
    """``ri_phys_footprint`` for one pid, or None if the pid is gone/unreadable."""
    buf = _RUsageInfoV4()
    rc = _libc.proc_pid_rusage(ctypes.c_int(int(pid)), ctypes.c_int(_RUSAGE_INFO_V4),
                               ctypes.byref(buf))
    if rc != 0:
        return None
    return int(buf.ri_phys_footprint)


def _child_map() -> dict[int, list[int]]:
    """ppid -> [pids] from a single `ps` snapshot."""
    out = subprocess.run(["/bin/ps", "-axo", "pid=,ppid="],
                         capture_output=True, text=True, timeout=15)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError("process-tree ps snapshot failed or was empty")
    kids: dict[int, list[int]] = {}
    for line in out.stdout.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 2:
            raise RuntimeError("malformed process-tree ps snapshot")
        try:
            pid, ppid = int(parts[0]), int(parts[1])
        except ValueError as exc:
            raise RuntimeError("malformed process-tree ps snapshot") from exc
        kids.setdefault(ppid, []).append(pid)
    return kids


def tree_pids(roots: list[int]) -> list[int]:
    """Every pid in the subtree(s) rooted at ``roots`` (roots included), pid-cycle safe."""
    kids = _child_map()
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(kids.get(pid, ()))
    return sorted(seen)


def tree_footprint(roots: list[int]) -> int:
    total = 0
    for pid in tree_pids(roots):
        fp = phys_footprint(pid)
        if fp is None:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue  # The process exited after the enumeration snapshot.
            except OSError as exc:
                raise RuntimeError(f"pid {pid} footprint and liveness unreadable") from exc
            raise RuntimeError(f"live pid {pid} phys_footprint unreadable")
        total += fp
    return total


def _footprint_p_bytes(pid: int) -> int | None:
    """Best-effort phys_footprint from `footprint -p <pid>` (for the self-check)."""
    try:
        out = subprocess.run(["footprint", "-p", str(pid)],
                             capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    import re
    for line in out.stdout.splitlines():
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*([KMGT]?B)", line)
        if m:
            v = float(m.group(1))
            # Apple's `footprint` prints BINARY units despite the "KB/MB/GB" labels
            # (verified: it renders 79.73 GiB as "80 GB", i.e. 1024-based, rounded to
            # whole units -- so this reader and footprint -p agree within rounding).
            mult = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}
            return int(v * mult.get(m.group(2), 1))
    return None


def main(argv: list[str]) -> int:
    if not argv:
        sys.stderr.write("usage: tree_footprint.py <root_pid>... | --pid P | --self-check P\n")
        return 2
    if argv[0] == "--pid":
        fp = phys_footprint(int(argv[1]))
        if fp is None:
            return 2
        print(fp)
        return 0
    if argv[0] == "--self-check":
        pid = int(argv[1])
        mine = phys_footprint(pid)
        fp_p = _footprint_p_bytes(pid)
        if mine is None:
            return 2
        print(f"{mine} footprint_p={fp_p if fp_p is not None else 'NA'}")
        return 0
    try:
        roots = [int(a) for a in argv]
    except ValueError:
        sys.stderr.write("root pids must be integers\n")
        return 2
    # An unreadable root is never a valid zero-sized tree. Descendant read failures
    # are checked against liveness by tree_footprint; only exited children are omitted.
    for r in roots:
        if phys_footprint(r) is None:
            sys.stderr.write(f"root pid {r} phys_footprint unreadable\n")
            return 2
    try:
        total = tree_footprint(roots)
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        sys.stderr.write(f"process-tree footprint unavailable: {exc}\n")
        return 2
    print(total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
