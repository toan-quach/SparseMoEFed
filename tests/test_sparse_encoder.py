"""Tests for the per-expert compressor + EF21 buffer behaviour."""

import numpy as np
import pytest
import torch

from sparsefedmoe.common.expert_compressor import (
    CompressionConfig,
    ExpertCompressor,
)


def _expert_name(layer, expert, proj="gate_proj"):
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"


def _make_deltas(num_layers=2, num_experts=4, dim=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    deltas = {}
    for l in range(num_layers):
        for e in range(num_experts):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                deltas[_expert_name(l, e, proj)] = torch.randn(dim, dim, generator=g) * 0.01
    deltas["model.layers.0.self_attn.q_proj.weight"] = torch.randn(dim, dim, generator=g) * 0.01
    return deltas


def _skewed_freq(num_layers=2, num_experts=4):
    freq = np.zeros((num_layers, num_experts))
    freq[:, 0] = 0.50   # HIGH
    freq[:, 1] = 0.02   # INT8 band (between 0.005 and 0.05)
    freq[:, 2] = 0.003  # SKIP
    freq[:, 3] = 0.001  # SKIP
    return freq


def test_tiers_assigned_correctly():
    cc = ExpertCompressor(CompressionConfig(skip_threshold=0.005, high_threshold=0.05))
    _, meta = cc.compress_expert_updates(_make_deltas(), _skewed_freq())
    cm = meta["compression_map"]
    assert cm[_expert_name(0, 0)] == "fp16"
    assert cm[_expert_name(0, 1)] == "int8"
    assert cm[_expert_name(0, 2)] == "skipped"
    assert cm[_expert_name(0, 3)] == "skipped"
    assert cm["model.layers.0.self_attn.q_proj.weight"] == "none"


def test_compression_ratio_lt_one():
    cc = ExpertCompressor(CompressionConfig(skip_threshold=0.005, high_threshold=0.05))
    _, meta = cc.compress_expert_updates(_make_deltas(), _skewed_freq())
    assert meta["bytes_compressed"] < meta["bytes_original"]
    assert meta["compression_ratio"] < 1.0


def test_int8_roundtrip_within_tolerance():
    original = torch.randn(64, 64) * 0.01
    packed = ExpertCompressor._quantize_int8(original)
    recon = ExpertCompressor._dequantize_int8(packed, original.shape)
    rel_err = (original - recon).abs().mean() / (original.abs().mean() + 1e-10)
    assert rel_err < 0.1


def test_error_feedback_accumulates_on_skip():
    cfg = CompressionConfig(skip_threshold=0.005, high_threshold=0.05, use_error_feedback=True)
    cc = ExpertCompressor(cfg)
    delta = {_expert_name(0, 0): torch.ones(8, 8) * 0.001}
    freq = np.array([[0.001]])  # SKIP
    cc.compress_expert_updates(delta, freq)
    key = (0, 0, _expert_name(0, 0))
    assert key in cc.error_buffers
    assert cc.error_buffers[key].shape == (8, 8)
    # Feed a second round of the same delta; buffer should now hold ≈2× the original.
    cc.compress_expert_updates(delta, freq)
    assert cc.error_buffers[key].mean().item() == pytest.approx(0.002, abs=1e-5)


def test_error_feedback_flush_promotes_to_int8():
    cfg = CompressionConfig(
        skip_threshold=0.005, high_threshold=0.05,
        use_error_feedback=True, ef_flush_threshold=0.01,
    )
    cc = ExpertCompressor(cfg)
    # Delta large enough that norm > 0.01 after a single round.
    delta = {_expert_name(0, 0): torch.ones(8, 8) * 0.5}
    freq = np.array([[0.001]])  # SKIP band
    compressed, meta = cc.compress_expert_updates(delta, freq)
    assert meta["compression_map"][_expert_name(0, 0)] == "int8"
    assert _expert_name(0, 0) in meta.get("ef_flushes", [])
    assert (0, 0, _expert_name(0, 0)) not in cc.error_buffers


def test_floor_promotes_skip_to_int8():
    cc = ExpertCompressor(CompressionConfig(skip_threshold=0.005, high_threshold=0.05))
    deltas = {_expert_name(0, 2): torch.ones(4, 4) * 0.001}
    freq = np.array([[0.0, 0.0, 0.001, 0.0]])  # expert 2 would normally SKIP
    _, meta = cc.compress_expert_updates(deltas, freq, floor_tiers={(0, 2): "INT8"})
    assert meta["compression_map"][_expert_name(0, 2)] == "int8"


def test_fp16_roundtrip():
    cc = ExpertCompressor(CompressionConfig(skip_threshold=0.005, high_threshold=0.05))
    name = _expert_name(0, 0)
    deltas = {name: torch.randn(16, 16) * 0.01}
    freq = np.array([[0.5]])
    compressed, meta = cc.compress_expert_updates(deltas, freq)
    decompressed = cc.decompress_expert_updates(compressed, meta)
    err = (deltas[name] - decompressed[name]).abs().max().item()
    assert err < 0.01


def test_decompress_skips_skipped_experts():
    cc = ExpertCompressor(CompressionConfig(skip_threshold=0.005, high_threshold=0.05))
    name = _expert_name(0, 0)
    deltas = {name: torch.ones(4, 4) * 0.001}
    freq = np.array([[0.001]])  # SKIP
    compressed, meta = cc.compress_expert_updates(deltas, freq)
    decompressed = cc.decompress_expert_updates(compressed, meta)
    assert name not in decompressed
