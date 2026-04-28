"""Tests for ActivationTracker against a synthetic MoE model."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparsefedmoe.client.activation_tracker import ActivationTracker


class _FakeSparseMoeBlock(torch.nn.Module):
    def __init__(self, dim: int, num_experts: int, fixed_logits=None):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.fixed_logits = fixed_logits

    def forward(self, x):
        bsz = x.shape[0]
        if self.fixed_logits is not None:
            logits = self.fixed_logits.expand(bsz, -1).to(x.device)
        else:
            logits = torch.zeros(bsz, self.num_experts)
        return x, logits


class _FakeMoEModel(torch.nn.Module):
    def __init__(self, num_layers=2, num_experts=4, top_k=2):
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_experts=num_experts,
            num_experts_per_tok=top_k,
        )
        # First layer: strongly route to expert 0; second: route to expert 2.
        logits1 = torch.zeros(1, num_experts); logits1[0, 0] = 10.0
        logits2 = torch.zeros(1, num_experts); logits2[0, 2] = 10.0
        self.layers = torch.nn.ModuleList([
            _FakeSparseMoeBlock(8, num_experts, fixed_logits=logits1),
            _FakeSparseMoeBlock(8, num_experts, fixed_logits=logits2),
        ])

    def forward(self, x):
        for l in self.layers:
            x, _ = l(x)
        return x


def test_tracker_counts_top_k_correctly():
    model = _FakeMoEModel(num_layers=2, num_experts=4, top_k=2)
    tracker = ActivationTracker(model)
    tracker.start()
    tokens = 16
    _ = model(torch.zeros(tokens, 8))
    tracker.stop()

    # Each token picks top-2 experts. In layer 0 the logit for expert 0 is 10
    # and the rest are 0, so top-2 = [0, any_other]. Expert 0 should be ~1.0,
    # and the remaining top-1 slot is split across experts 1..3 (tiebreak is
    # torch's choice).
    profile = tracker.get_activation_profile()
    assert profile.shape == (2, 4)
    # Arch §3.1 invariant: Σ_e freq[l][e] = top_k per layer.
    for l in range(2):
        assert profile[l].sum() == pytest.approx(2.0, abs=1e-6)
    # Expert 0 is the top-1 choice for every token in layer 0 → freq = 1.0.
    assert profile[0, 0] == pytest.approx(1.0, abs=1e-6)
    # Expert 2 is the top-1 choice for every token in layer 1.
    assert profile[1, 2] == pytest.approx(1.0, abs=1e-6)


def test_metadata_round_trip():
    model = _FakeMoEModel()
    tracker = ActivationTracker(model)
    tracker.start()
    _ = model(torch.zeros(4, 8))
    tracker.stop()
    md = tracker.get_activation_metadata()
    assert set(md) >= {"activation_freq", "total_tokens", "num_layers", "num_experts", "top_k"}
    arr = np.asarray(md["activation_freq"])
    assert arr.shape == (model.config.num_hidden_layers, model.config.num_experts)


def test_reset_zeroes_counts():
    model = _FakeMoEModel()
    tracker = ActivationTracker(model)
    tracker.start()
    _ = model(torch.zeros(4, 8))
    tracker.stop()
    assert tracker.total_tokens > 0
    tracker.reset()
    assert tracker.total_tokens == 0
    assert tracker.expert_counts.sum() == 0
