"""Tests for the new per-round metrics: router z-loss, validation eval, and
communication-cost accounting.

The server-side helpers are importable without NVFlare because they're at
module scope in ``freq_weighted_controller`` and only touch ``.params`` /
``.meta`` via attribute access — we feed them ``SimpleNamespace`` stand-ins.
"""

from types import SimpleNamespace
from typing import Dict

import numpy as np
import pytest
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Trainer-side helpers: z-loss + evaluate()
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def _trainer_module():
    """Import the trainer without requiring nvflare to be installed.

    The trainer imports ``nvflare.client`` and ``nvflare.app_common...`` at
    module top level; stub them so tests can run in a CPU-only env.
    """
    import sys, types
    for stub in [
        "nvflare", "nvflare.client", "nvflare.app_common",
        "nvflare.app_common.abstract", "nvflare.app_common.abstract.fl_model",
    ]:
        sys.modules.setdefault(stub, types.ModuleType(stub))
    fl_mod = sys.modules["nvflare.app_common.abstract.fl_model"]

    class _FLModel:  # minimal shape used by the trainer
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _PT:
        DIFF = "DIFF"
        WEIGHTS = "WEIGHTS"

    fl_mod.FLModel = _FLModel
    fl_mod.ParamsType = _PT
    cl = sys.modules["nvflare.client"]
    cl.init = lambda: None
    cl.get_site_name = lambda: "site-0"
    cl.is_running = lambda: False
    cl.receive = lambda: None
    cl.send = lambda x: None

    import importlib
    mod = importlib.import_module("sparsefedmoe.client.olmoe_sft_trainer")
    return mod


class TestRouterZLoss:
    def test_returns_none_for_none_input(self, _trainer_module):
        assert _trainer_module._compute_router_z_loss(None) is None

    def test_returns_none_for_empty_tuple(self, _trainer_module):
        assert _trainer_module._compute_router_z_loss(tuple()) is None

    def test_accepts_single_tensor(self, _trainer_module):
        logits = torch.zeros(4, 8)  # uniform logits
        z = _trainer_module._compute_router_z_loss(logits)
        assert z is not None
        # logsumexp(0s over 8 experts) = log(8); squared = log(8)^2
        assert abs(float(z) - (np.log(8) ** 2)) < 1e-5

    def test_averages_across_layers(self, _trainer_module):
        # Per-layer tensors, matches HF format.
        l1 = torch.zeros(4, 8)  # logsumexp -> log(8)
        l2 = torch.full((4, 8), float("-inf"))
        l2[:, 0] = 0.0            # logsumexp -> 0
        z = _trainer_module._compute_router_z_loss((l1, l2))
        expected = 0.5 * (np.log(8) ** 2 + 0.0)
        assert abs(float(z) - expected) < 1e-5

    def test_ignores_non_tensor_entries(self, _trainer_module):
        l1 = torch.zeros(2, 4)
        z = _trainer_module._compute_router_z_loss((l1, None, "junk"))
        assert z is not None


class TestEvaluate:
    """``evaluate()`` is token-weighted, uses no-grad, and flips the model back
    to train mode on exit."""

    def _tiny_model(self):
        # Minimal language-model-shape stub. Returns a constant loss per batch
        # so we can check the token-weighted averaging explicitly.
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # a trainable param so model.train()/.eval() transitions are observable
                self.dummy = torch.nn.Parameter(torch.zeros(1))
                self.last_mode_seen = None

            def forward(self, input_ids=None, labels=None, attention_mask=None, **_):
                self.last_mode_seen = self.training
                # Loss per-batch equals number of non-ignored tokens / 10,
                # so loss * n_tokens is easy to predict.
                n = int((labels != -100).sum().item())
                return SimpleNamespace(loss=torch.tensor(n / 10.0, requires_grad=True))

        return Stub()

    def test_returns_none_when_dataloader_is_none(self, _trainer_module):
        out = _trainer_module.evaluate(self._tiny_model(), None)
        assert out is None

    def test_token_weighted_average(self, _trainer_module, monkeypatch):
        monkeypatch.setattr(_trainer_module, "DEVICE", "cpu")
        model = self._tiny_model()
        # Two batches: one with 5 real tokens (loss 0.5), one with 10 (loss 1.0).
        # Token-weighted mean = (0.5*5 + 1.0*10) / 15 = 12.5 / 15 ≈ 0.8333
        batch_a = {
            "input_ids": torch.zeros(1, 5, dtype=torch.long),
            "labels": torch.zeros(1, 5, dtype=torch.long),
        }
        batch_b = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "labels": torch.zeros(1, 10, dtype=torch.long),
        }
        out = _trainer_module.evaluate(model, [batch_a, batch_b])
        assert out is not None
        assert abs(out["eval_loss"] - 12.5 / 15) < 1e-5
        assert abs(out["perplexity"] - np.exp(12.5 / 15)) < 1e-4
        assert out["eval_tokens"] == 15
        # Model should end in train mode (evaluate toggles back).
        assert model.training is True
        # And have run in eval mode inside.
        assert model.last_mode_seen is False

    def test_attention_mask_does_not_affect_token_count(self, _trainer_module, monkeypatch):
        monkeypatch.setattr(_trainer_module, "DEVICE", "cpu")
        model = self._tiny_model()
        # 4 positions, attention_mask marks only 2 as real — but evaluate()
        # counts tokens via labels != -100 (matching the model's CE reduction),
        # so all 4 count.
        batch = {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "labels": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        }
        out = _trainer_module.evaluate(model, [batch])
        # n_tokens = 4 (all labels != -100). Model loss = 4/10 = 0.4.
        # eval_loss = (0.4 * 4) / 4 = 0.4.
        assert out["eval_tokens"] == 4
        assert abs(out["eval_loss"] - 0.4) < 1e-5

    def test_perplexity_clamped_on_huge_loss(self, _trainer_module, monkeypatch):
        monkeypatch.setattr(_trainer_module, "DEVICE", "cpu")

        class HugeLossModel(torch.nn.Module):
            def forward(self, input_ids=None, labels=None, **_):
                return SimpleNamespace(loss=torch.tensor(1e6, requires_grad=True))

        batch = {
            "input_ids": torch.zeros(1, 2, dtype=torch.long),
            "labels": torch.zeros(1, 2, dtype=torch.long),
        }
        out = _trainer_module.evaluate(HugeLossModel(), [batch])
        # Perplexity should be finite (we clamp exp argument at 50).
        assert np.isfinite(out["perplexity"])
        assert out["perplexity"] == pytest.approx(np.exp(50.0))


# ──────────────────────────────────────────────────────────────────────────────
# Server-side helpers: bytes accounting + eval aggregation
# ──────────────────────────────────────────────────────────────────────────────
# The controller module pulls in nvflare at import time to subclass
# ``ModelController``. The helpers we want to test are pure-numpy and don't
# actually use nvflare, so we stub the imports and load the module anyway —
# this way the tests run in any CPU-only env.

def _load_server_helpers():
    import sys
    import types

    for name in [
        "nvflare",
        "nvflare.apis",
        "nvflare.apis.fl_context",
        "nvflare.app_common",
        "nvflare.app_common.abstract",
        "nvflare.app_common.abstract.fl_model",
        "nvflare.app_common.workflows",
        "nvflare.app_common.workflows.model_controller",
    ]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["nvflare.apis.fl_context"].FLContext = object

    class _FLModel:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    sys.modules["nvflare.app_common.abstract.fl_model"].FLModel = _FLModel

    class _ModelController:
        def __init__(self, *a, **kw): pass

    sys.modules["nvflare.app_common.workflows.model_controller"].ModelController = (
        _ModelController
    )

    import importlib
    mod = importlib.import_module(
        "sparsefedmoe.server.freq_weighted_controller"
    )
    return mod


_server = _load_server_helpers()
_aggregate_eval_metrics = _server._aggregate_eval_metrics
_estimate_params_bytes = _server._estimate_params_bytes
_fmt_bytes = _server._fmt_bytes
_sum_uplink_bytes = _server._sum_uplink_bytes


def _mk_result(params=None, meta=None):
    return SimpleNamespace(params=params or {}, meta=meta or {})


class TestEstimateParamsBytes:
    def test_empty_returns_zero(self):
        assert _estimate_params_bytes({}) == 0
        assert _estimate_params_bytes(None) == 0

    def test_ndarray_uses_nbytes(self):
        params = {"a": np.zeros(10, dtype=np.float32)}  # 40 bytes
        assert _estimate_params_bytes(params) == 40

    def test_mixed_dtypes(self):
        params = {
            "a": np.zeros(10, dtype=np.float32),  # 40
            "b": np.zeros(10, dtype=np.float16),  # 20
        }
        assert _estimate_params_bytes(params) == 60

    def test_int8_dict_counts_quantized_plus_scale(self):
        params = {"e": {"quantized": np.zeros(100, dtype=np.int8), "scale": 0.5}}
        # 100 bytes of int8 + 4 bytes of scale.
        assert _estimate_params_bytes(params) == 104


class TestSumUplinkBytes:
    def test_no_compression_metadata_counts_raw_params(self):
        r = _mk_result(params={"w": np.zeros(10, dtype=np.float32)})
        assert _sum_uplink_bytes({"c0": r}) == 40

    def test_compressed_uses_bytes_compressed_for_experts(self):
        # One expert param (compressed -> 50 bytes) and one shared param
        # (40 bytes ndarray). Total should be 90, not 40 + raw expert size.
        expert = {"quantized": np.zeros(46, dtype=np.int8), "scale": 0.1}
        r = _mk_result(
            params={
                "model.layers.0.mlp.experts.0.gate_proj.weight": expert,
                "model.layers.0.self_attn.q_proj.weight": np.zeros(10, dtype=np.float32),
            },
            meta={
                "compression_metadata": {
                    "bytes_compressed": 50,
                    "compression_map": {
                        "model.layers.0.mlp.experts.0.gate_proj.weight": "int8",
                    },
                },
            },
        )
        assert _sum_uplink_bytes({"c0": r}) == 50 + 40

    def test_skipped_expert_contributes_zero_from_params_dict(self):
        # Skipped experts aren't in params at all; compression_map still lists
        # them as "skipped" but bytes_compressed should already account for
        # that (i.e. skipped contributes 0).
        r = _mk_result(
            params={"shared.norm.weight": np.zeros(4, dtype=np.float32)},  # 16
            meta={
                "compression_metadata": {
                    "bytes_compressed": 0,  # only a skipped expert
                    "compression_map": {
                        "model.layers.0.mlp.experts.3.gate_proj.weight": "skipped",
                    },
                },
            },
        )
        assert _sum_uplink_bytes({"c0": r}) == 16

    def test_sums_across_clients(self):
        r1 = _mk_result(params={"w": np.zeros(10, dtype=np.float32)})
        r2 = _mk_result(params={"w": np.zeros(20, dtype=np.float32)})
        assert _sum_uplink_bytes({"c0": r1, "c1": r2}) == 40 + 80


class TestAggregateEvalMetrics:
    def test_returns_none_when_no_client_reports(self):
        r = _mk_result(meta={})
        assert _aggregate_eval_metrics({"c0": r}) is None

    def test_token_weighted_across_clients(self):
        # c0: loss=1.0, tokens=100 ; c1: loss=2.0, tokens=300
        # weighted mean = (100 + 600) / 400 = 1.75
        r0 = _mk_result(meta={"eval_loss": 1.0, "eval_tokens": 100})
        r1 = _mk_result(meta={"eval_loss": 2.0, "eval_tokens": 300})
        out = _aggregate_eval_metrics({"c0": r0, "c1": r1})
        assert out is not None
        assert abs(out["eval_loss"] - 1.75) < 1e-9
        assert abs(out["perplexity"] - np.exp(1.75)) < 1e-5
        assert out["tokens"] == 400
        assert out["clients"] == 2

    def test_zero_token_client_is_ignored(self):
        r0 = _mk_result(meta={"eval_loss": 99.0, "eval_tokens": 0})
        r1 = _mk_result(meta={"eval_loss": 2.0, "eval_tokens": 10})
        out = _aggregate_eval_metrics({"c0": r0, "c1": r1})
        assert out["clients"] == 1
        assert abs(out["eval_loss"] - 2.0) < 1e-9


class TestFmtBytes:
    @pytest.mark.parametrize("n,expected_unit", [
        (0, "B"),
        (500, "B"),
        (2048, "KB"),
        (5 * 1024 * 1024, "MB"),
        (3 * 1024 ** 3, "GB"),
    ])
    def test_picks_right_unit(self, n, expected_unit):
        assert _fmt_bytes(n).endswith(expected_unit)
