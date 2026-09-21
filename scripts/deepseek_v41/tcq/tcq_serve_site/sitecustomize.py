"""F39 serve site hook: arm the tcq3 lane in the `mtplx serve` process at interpreter startup.

Python auto-imports ``sitecustomize`` from any directory on ``sys.path``.  The eval driver puts THIS directory on
the served process's PYTHONPATH (alongside the F39 ``tcq`` package) only for a ``--bank tcq3`` run, so:
  * when MTPLX_DSV41_TCQ3 != 1 (the mxfp4 control, or any other serve), this is a pure no-op — the stock decode
    and loader are untouched;
  * when MTPLX_DSV41_TCQ3 == 1, it installs the tcq3 loader (manifest + spec codec/record_bytes) and the tcq3
    serve decode (rebinds _dispatch_component_bank) BEFORE the model loads, so the served DeepSeek-V4.1 decodes the
    tcq3 bank through the F35 tile kernel.

Failures are surfaced loudly (the whole point is that a tcq3 serve must not silently fall back to a codec it cannot
decode): a broken install raises, so the guarded window fails fast rather than serving garbage.
"""
import os

if os.environ.get("MTPLX_DSV41_TCQ3") == "1":
    import tcq.loader_install as _loader
    import tcq.serve_install as _serve

    _loader.install_tcq_loader()          # manifest mode==tcq3 + ExpertStreamingModelSpec codec/record_bytes
    _serve.install_from_env()             # rebinds _dispatch_component_bank for tcq3 switches (env-gated)
    print("[tcq3-serve-site] armed: tcq3 loader + serve decode installed (MTPLX_DSV41_TCQ3=1)", flush=True)
