"""P1.0: ModelArgs config load + the §0 CSA2 per-layer mode table."""

from __future__ import annotations

import json
import os

import pytest

from mtplx.models.deepseek_v41 import (
    MODE_FULL,
    MODE_REINDEX,
    MODE_REUSE,
    MODE_SWA_ONLY,
    ModelArgs,
)

_REAL_CONFIG = os.environ.get(
    "MTPLX_DSV41_CONFIG",
    os.path.expanduser(
        "~/models/DeepSeek-V4.1-Flash-MTPLX-streaming-q2/config.json"
    ),
)


def _expected_modes() -> list[str]:
    """The §0 table written out per layer for the released 40-layer config."""
    modes = [MODE_SWA_ONLY, MODE_SWA_ONLY]  # 0, 1
    for _ in (2, 8, 14):  # Full then five Reuse (serves the six-layer ratio-2 run)
        modes.append(MODE_FULL)
        modes.extend([MODE_REUSE] * 5)
    modes.append(MODE_FULL)  # 20: CED boundary, then only 21-23 reuse it
    modes.extend([MODE_REUSE] * 3)
    for _ in (24, 28, 32, 36):  # Reindex then three Reuse
        modes.append(MODE_REINDEX)
        modes.extend([MODE_REUSE] * 3)
    return modes


@pytest.mark.skipif(
    not os.path.exists(_REAL_CONFIG), reason=f"artifact config missing: {_REAL_CONFIG}"
)
def test_layer_modes():
    args = ModelArgs.from_dict(json.load(open(_REAL_CONFIG)))

    assert args.model_type == "deepseek_v41"
    assert args.num_hidden_layers == 40
    assert args.kv_source_layer_ids == [2, 8, 14, 20]
    assert args.index_source_layer_ids == [2, 8, 14, 20, 24, 28, 32, 36]
    assert args.candidate_source_layer_id == 20
    assert len(args.compress_ratios) == 43  # 40 text + 3 MTP, MTP entries unused

    expected = _expected_modes()
    assert len(args.layer_modes) == 40
    assert args.layer_modes == expected, [
        (i, a, b)
        for i, (a, b) in enumerate(zip(args.layer_modes, expected))
        if a != b
    ]

    # Spot-checks of the §0 prose, independent of the loop above.
    assert args.layer_modes[0] == MODE_SWA_ONLY
    assert args.layer_modes[2] == MODE_FULL
    assert args.layer_modes[7] == MODE_REUSE
    assert args.layer_modes[20] == MODE_FULL
    assert args.layer_modes[24] == MODE_REINDEX
    assert args.layer_modes[39] == MODE_REUSE

    counts = {m: args.layer_modes.count(m) for m in set(args.layer_modes)}
    assert counts[MODE_SWA_ONLY] == 2
    assert counts[MODE_FULL] == 4
    assert counts[MODE_REINDEX] == 4
    assert counts[MODE_REUSE] == 30


def test_from_dict_accepts_bare_text_config():
    """`from_dict` takes either the nested or the flattened config."""
    full = json.load(open(_REAL_CONFIG)) if os.path.exists(_REAL_CONFIG) else None
    if full is None:
        pytest.skip("artifact config missing")
    bare = full["text_config"]
    assert ModelArgs.from_dict(full).layer_modes == ModelArgs.from_dict(bare).layer_modes
