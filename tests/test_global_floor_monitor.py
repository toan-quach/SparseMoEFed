"""Tests for GlobalFloorMonitor (arch §3.5)."""

import numpy as np
import pytest

from sparsefedmoe.server.global_floor_monitor import GlobalFloorMonitor


def test_starved_expert_enters_floor_set():
    # 2 layers, 4 experts. Expert 3 in each layer never gets activation.
    profile_a = np.array([
        [0.40, 0.30, 0.30, 0.001],
        [0.50, 0.25, 0.25, 0.001],
    ])
    profile_b = np.array([
        [0.35, 0.35, 0.30, 0.002],
        [0.45, 0.30, 0.25, 0.001],
    ])
    fm = GlobalFloorMonitor(floor_gamma=0.10, min_clients=2)
    floor = fm.update({"c0": profile_a, "c1": profile_b})
    # Expert 3 in both layers should be protected.
    assert (0, 3) in floor
    assert (1, 3) in floor
    # Popular experts should not.
    assert (0, 0) not in floor
    assert (1, 0) not in floor


def test_below_min_clients_keeps_prior_floor():
    fm = GlobalFloorMonitor(floor_gamma=0.10, min_clients=2)
    p1 = np.array([[0.5, 0.5, 0.001]])
    p2 = np.array([[0.5, 0.5, 0.001]])
    first = fm.update({"c0": p1, "c1": p2})
    assert (0, 2) in first
    # Single-client update does not change the floor.
    fm.update({"c0": p1})
    assert fm.get_floor_tiers() == first


def test_floor_tier_list_is_serialisable():
    fm = GlobalFloorMonitor(floor_gamma=0.10, min_clients=2)
    profile = np.array([[0.5, 0.5, 0.001, 0.002]])
    fm.update({"c0": profile, "c1": profile})
    lst = fm.get_floor_tier_list()
    assert isinstance(lst, list)
    for entry in lst:
        assert len(entry) == 3
        l, e, tier = entry
        assert isinstance(l, int) and isinstance(e, int)
        assert tier == "INT8"


def test_layer_scoped_thresholds():
    # Layer 0 is nearly uniform; layer 1 is extremely concentrated. An
    # expert at 0.10 in layer 0 should NOT be floored; an expert at 0.05 in
    # layer 1 (where mean is huge) SHOULD.
    p1 = np.array([
        [0.30, 0.30, 0.30, 0.10],
        [10.0, 0.05, 0.05, 0.05],
    ])
    p2 = np.array([
        [0.30, 0.30, 0.30, 0.10],
        [10.0, 0.05, 0.05, 0.05],
    ])
    fm = GlobalFloorMonitor(floor_gamma=0.10, min_clients=2)
    floor = fm.update({"c0": p1, "c1": p2})
    assert (0, 3) not in floor  # 0.10 is well above 0.10 * mean(0.25) = 0.025
    assert (1, 1) in floor      # 0.05 is below 0.10 * mean(2.54) = 0.254
