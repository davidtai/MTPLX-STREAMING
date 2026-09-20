"""Pin MLX to CPU BEFORE any import that pulls in mlx.core, and put the F6
helper dir on sys.path so `import engram_parallel` / `import stage_f6_runner`
resolve. Importing NGramRowCache (transitively, via engram_parallel) imports
mlx.core, which defaults to Metal; these are CPU-only tests and no Metal work is
permitted, so the device is forced to CPU here first.
"""
import sys
from pathlib import Path

import mlx.core as mx
import pytest

mx.set_default_device(mx.cpu)

_F6_DIR = str(Path(__file__).resolve().parent.parent)
if _F6_DIR not in sys.path:
    sys.path.insert(0, _F6_DIR)


@pytest.fixture(autouse=True)
def _reset_f6_states():
    """Shut down and clear the module-level install registry between tests so
    stats() is per-test and no thread pools accumulate across the session."""
    import engram_parallel as ep

    def _clear():
        for st in list(ep._STATES):
            st.shutdown()
        ep._STATES.clear()

    _clear()
    yield
    _clear()
