"""Tests for the freq-weighted aggregation math (arch §3.3) + router alignment (§3.4).

We exercise ``FreqWeightedFedAvg._aggregate`` directly with hand-built FLModel
results, so these tests don't require an NVFlare server running.
"""

import numpy as np
import pytest

# Import will only succeed if nvflare is installed. Skip cleanly otherwise so
# this test file still runs in envs without nvflare (e.g., pure-math CI).
nvflare = pytest.importorskip("nvflare")
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType  # noqa: E402

from sparsefedmoe.server.freq_weighted_controller import FreqWeightedFedAvg  # noqa: E402


def _mk_profile(num_experts, layer_freqs):
    """layer_freqs: list of per-layer dicts {expert_idx: freq}. Fill rest with 0."""
    out = np.zeros((len(layer_freqs), num_experts))
    for l, row in enumerate(layer_freqs):
        for e, f in row.items():
            out[l, e] = f
    return out


def _mk_result(params, profile, tokens=100, skipped=None):
    return FLModel(
        params_type=ParamsType.DIFF,
        params=params,
        meta={
            "activation_profile": {
                "activation_freq": profile.tolist(),
                "total_tokens": tokens,
                "num_layers": profile.shape[0],
                "num_experts": profile.shape[1],
                "top_k": 2,
            },
            "skipped_experts": list(skipped or []),
        },
    )


def test_freq_weighted_mean_matches_hand_calculation():
    """Two clients both update expert (0, 0) with different frequencies."""
    ctrl = FreqWeightedFedAvg(num_clients=2, num_rounds=1, min_clients=1)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    shape = (4, 4)

    a = np.full(shape, 1.0, dtype=np.float32)
    b = np.full(shape, 5.0, dtype=np.float32)

    prof_a = _mk_profile(4, [{0: 0.75, 1: 0.25}])
    prof_b = _mk_profile(4, [{0: 0.25, 1: 0.75}])

    results = {
        "c0": _mk_result({name: a}, prof_a),
        "c1": _mk_result({name: b}, prof_b),
    }
    out = ctrl._aggregate(results, global_params={name: np.zeros(shape, dtype=np.float32)})
    # Expected: 0.75/(0.75+0.25) * 1 + 0.25/(0.75+0.25) * 5 = 0.75 + 1.25 = 2.0
    np.testing.assert_allclose(out[name], np.full(shape, 2.0, dtype=np.float32), atol=1e-6)


def test_no_contributor_keeps_global_params():
    ctrl = FreqWeightedFedAvg(num_clients=2, num_rounds=1, min_clients=1)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    shape = (4, 4)
    other = "model.layers.0.mlp.experts.1.gate_proj.weight"
    a = np.full(shape, 1.0, dtype=np.float32)
    b = np.full(shape, 5.0, dtype=np.float32)

    # Only supply updates for "other". Aggregation for `name` should fall back.
    prof = _mk_profile(4, [{0: 0.1, 1: 0.1}])
    results = {
        "c0": _mk_result({other: a}, prof),
        "c1": _mk_result({other: b}, prof),
    }
    global_params = {name: np.full(shape, 42.0, dtype=np.float32)}
    out = ctrl._aggregate(results, global_params=global_params)
    np.testing.assert_array_equal(out[name], np.full(shape, 42.0, dtype=np.float32))


def test_skipped_experts_excluded_from_aggregation():
    ctrl = FreqWeightedFedAvg(num_clients=2, num_rounds=1, min_clients=1)
    name = "model.layers.0.mlp.experts.0.gate_proj.weight"
    shape = (4, 4)
    a = np.full(shape, 100.0, dtype=np.float32)
    b = np.full(shape, 1.0, dtype=np.float32)

    prof = _mk_profile(4, [{0: 0.5}])
    # c0 lists the param as skipped even though the value is in params (edge case).
    results = {
        "c0": _mk_result({name: a}, prof, skipped=[name]),
        "c1": _mk_result({name: b}, prof),
    }
    out = ctrl._aggregate(results, global_params={name: np.zeros(shape, dtype=np.float32)})
    # Only c1 contributes.
    np.testing.assert_allclose(out[name], b, atol=1e-6)


def test_shared_params_dataset_size_weighted():
    ctrl = FreqWeightedFedAvg(num_clients=2, num_rounds=1, min_clients=1)
    shared_name = "model.layers.0.self_attn.q_proj.weight"
    shape = (4, 4)
    a = np.full(shape, 10.0, dtype=np.float32)
    b = np.full(shape, 20.0, dtype=np.float32)

    prof = _mk_profile(4, [{0: 0.5}])
    results = {
        "c0": _mk_result({shared_name: a}, prof, tokens=100),
        "c1": _mk_result({shared_name: b}, prof, tokens=300),
    }
    out = ctrl._aggregate(results, global_params={shared_name: np.zeros(shape, dtype=np.float32)})
    # (100*10 + 300*20) / 400 = 17.5
    np.testing.assert_allclose(out[shared_name], np.full(shape, 17.5, dtype=np.float32), atol=1e-6)
