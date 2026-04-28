"""Tests for alignment-weighted router aggregation (arch §3.4)."""

import numpy as np
import pytest

from sparsefedmoe.server.router_alignment import compute_router_weights


def test_single_client_gets_full_weight():
    profile = np.array([[0.5, 0.5]])
    weights, entropy = compute_router_weights({"c0": profile}, tau_active=0.01)
    assert weights == {"c0": 1.0}
    assert entropy == 0.0


def test_popular_client_outweighs_maverick():
    # c_popular and c_shared both use experts 0,1 (popular across federation).
    # c_maverick uses expert 2 alone.
    shared = np.array([[0.5, 0.5, 0.0]])
    maverick = np.array([[0.0, 0.0, 0.5]])
    weights, entropy = compute_router_weights({
        "c_popular": shared,
        "c_shared": shared,
        "c_maverick": maverick,
    }, tau_active=0.01)
    assert weights["c_popular"] > weights["c_maverick"]
    assert weights["c_shared"] > weights["c_maverick"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert entropy > 0


def test_identical_profiles_uniform_weights():
    profile = np.array([[0.25, 0.25, 0.25, 0.25]])
    weights, _ = compute_router_weights(
        {"a": profile, "b": profile, "c": profile}, tau_active=0.01,
    )
    for w in weights.values():
        assert w == pytest.approx(1 / 3, abs=1e-9)


def test_all_zero_alignment_falls_back_to_uniform():
    # tau_active too high → no active experts at all.
    profile = np.array([[0.01, 0.01, 0.01]])
    weights, _ = compute_router_weights(
        {"a": profile, "b": profile}, tau_active=0.99,
    )
    assert weights == {"a": 0.5, "b": 0.5}
