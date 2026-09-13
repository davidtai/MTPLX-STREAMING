"""W97 review fix (item 2): the f32 ``wo_a`` cache is priced into the memory plan.

The cache is materialised lazily at the first decode token -- after
``build_streaming_config`` and the W62 budget derivation have fixed the plan --
so ``derived_expert_cache_allowance_bytes`` (which subtracts ``plan.fixed_bytes``)
would otherwise hand the streamed expert cache its full allowance and the process
would run ~5.4 GB over plan.  These tests prove the loader reserves exactly
``NUM_TEXT_LAYERS * WO_A_DENSE_F32_BYTES`` as fixed ``additional_resident_bytes``
when the lever is armed, and nothing when it is off.

Pure arithmetic on the loader helper -- no model build, no MLX device work; CPU
pinned before the model import chain runs (worker-tests-must-pin-mlx-cpu.md).
"""

from __future__ import annotations

import mlx.core as mx  # the loader helper imports deepseek_v41 (imports mlx.core)

mx.set_default_device(mx.cpu)

import pytest

from mtplx.models import deepseek_v41 as dsv41
from mtplx.models import deepseek_v41_loader as loader


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(dsv41._ATTN_WO_A_CACHE_ENV, raising=False)
    monkeypatch.delenv(dsv41._ATTN_FUSED_PROJ_ENV, raising=False)
    yield


def test_wo_a_dense_bytes_matches_released_dims():
    # wo_a = [o_groups*o_lora_rank, n_heads*head_dim//o_groups] = [8192, 4096], f32
    assert loader.WO_A_DENSE_ROWS == 8192          # 8 groups * 1024 o_lora_rank
    assert loader.WO_A_DENSE_COLS == 4096          # 64 heads * 512 head_dim // 8
    assert loader.WO_A_DENSE_F32_BYTES == 8192 * 4096 * 4
    assert loader.WO_A_CACHE_RESIDENT_BYTES == loader.NUM_TEXT_LAYERS * loader.WO_A_DENSE_F32_BYTES
    # sanity: ~5.4 GB across the 40 backbone layers (both codecs -- f32 either way)
    assert 5.0e9 < loader.WO_A_CACHE_RESIDENT_BYTES < 5.8e9


def test_wo_a_cache_priced_into_additional_resident(monkeypatch):
    monkeypatch.setenv(dsv41._ATTN_WO_A_CACHE_ENV, "0")
    off = loader.deepseek_v41_additional_resident_bytes()

    monkeypatch.setenv(dsv41._ATTN_WO_A_CACHE_ENV, "1")
    on = loader.deepseek_v41_additional_resident_bytes()

    # Off: SWA window only.  On: SWA window + the f32 wo_a cache reserve.
    assert off == loader.SWA_WINDOW_BYTES
    assert on == loader.SWA_WINDOW_BYTES + loader.WO_A_CACHE_RESIDENT_BYTES
    # The whole point: armed - off == n_layers * WO_A_DENSE_F32_BYTES.
    assert on - off == loader.NUM_TEXT_LAYERS * loader.WO_A_DENSE_F32_BYTES


def test_additional_resident_defaults_to_swa_only(monkeypatch):
    # Lever unset (the shipped default): reserve is exactly the SWA window, so an
    # off run prices no wo_a bytes (the cache is never built).
    monkeypatch.delenv(dsv41._ATTN_WO_A_CACHE_ENV, raising=False)
    assert loader.deepseek_v41_additional_resident_bytes() == loader.SWA_WINDOW_BYTES
