"""Tests for parameter-name classification helpers."""

import pytest

from sparsefedmoe.common.model_utils import (
    classify_param,
    is_expert_param,
    is_router_param,
    parse_expert_indices,
)


@pytest.mark.parametrize("name,expected", [
    ("model.layers.5.mlp.experts.23.gate_proj.weight", (5, 23)),
    ("model.layers.0.mlp.experts.0.down_proj.weight", (0, 0)),
    ("model.layers.7.self_attn.q_proj.weight", (None, None)),
    ("model.embed_tokens.weight", (None, None)),
    ("model.layers.1.mlp.gate.weight", (None, None)),
    ("base_model.model.model.layers.2.mlp.experts.3.up_proj.lora_A.default.weight", (2, 3)),
])
def test_parse_expert_indices(name, expected):
    assert parse_expert_indices(name) == expected


@pytest.mark.parametrize("name,expert,router", [
    ("model.layers.5.mlp.experts.23.gate_proj.weight", True, False),
    ("model.layers.5.mlp.gate.weight", False, True),
    ("model.layers.5.self_attn.q_proj.weight", False, False),
    ("model.layers.0.mlp.experts.0.down_proj.weight", True, False),
])
def test_classification(name, expert, router):
    assert is_expert_param(name) == expert
    assert is_router_param(name) == router
    expected_class = "expert" if expert else "router" if router else "shared"
    assert classify_param(name) == expected_class
