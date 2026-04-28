"""FreqWeightedFedAvg — NVFlare ModelController for SparseFedMoE (arch §3.3).

Aggregation rules:
  - Expert FFN params:        frequency-weighted over the clients that transmitted
                              an update for that expert (§3.3).
  - Router (gate) params:     alignment-weighted (§3.4) when enabled.
  - Everything else (shared): dataset-size-weighted FedAvg.

Additionally, between rounds the controller runs the Global Floor Monitor
(§3.5) over the incoming activation profiles and attaches ``floor_tiers`` to
the next round's outgoing FLModel metadata so the client encoder can promote
under-utilised experts from SKIP to INT8.

Pure-numpy math — does not touch torch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.workflows.model_controller import ModelController

from sparsefedmoe.common.config import SparseFedMoEConfig
from sparsefedmoe.common.model_utils import (
    classify_param,
    parse_expert_indices,
)
from sparsefedmoe.server.global_floor_monitor import GlobalFloorMonitor
from sparsefedmoe.server.router_alignment import compute_router_weights

logger = logging.getLogger(__name__)


class FreqWeightedFedAvg(ModelController):
    def __init__(
        self,
        num_clients: int = 5,
        num_rounds: int = 50,
        min_clients: int = 2,
        persistor_id: str = "persistor",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.min_clients = min_clients
        self.persistor_id = persistor_id
        self.cfg = SparseFedMoEConfig.from_dict(config or {})

        self.floor_monitor = GlobalFloorMonitor(
            floor_gamma=self.cfg.floor_gamma,
            min_clients=self.cfg.floor_min_clients,
        ) if self.cfg.use_floor else None

        # For diagnostics.
        self.client_profiles: Dict[str, np.ndarray] = {}

        # Cumulative communication accounting (arch §3 motivation).
        # Uplink  = bytes sent by clients to server (compressed deltas).
        # Downlink = bytes broadcast from server to clients (global model).
        # We reset per-round counters each round and keep running totals.
        self.cumulative_uplink_bytes: int = 0
        self.cumulative_downlink_bytes: int = 0

    # ── Main control flow ───────────────────────────────────────────────────
    def run(self, fl_ctx: FLContext):  # type: ignore[override]
        model = self.load_model()
        logger.info("Loaded initial global model: %d params", len(model.params))

        for round_idx in range(self.num_rounds):
            logger.info("── Round %d/%d ──", round_idx + 1, self.num_rounds)

            # Attach floor tiers to outgoing metadata (if any).
            if self.floor_monitor is not None:
                floor_list = self.floor_monitor.get_floor_tier_list()
                if model.meta is None:
                    model.meta = {}
                model.meta["floor_tiers"] = floor_list
                if floor_list:
                    logger.info("Broadcast floor tiers: %d experts protected", len(floor_list))

            # ── Downlink bookkeeping ──
            # One per-client-sized copy goes to each expected participant. We
            # use ``num_clients`` (the federation size the controller was
            # configured for) rather than the actual response count, because
            # the broadcast happens before we know who will respond and the
            # unresponsive-client case still pays the outbound cost.
            downlink_per_client = _estimate_params_bytes(model.params)
            round_downlink = downlink_per_client * self.num_clients
            self.cumulative_downlink_bytes += round_downlink

            results = self.send_model_and_wait(
                targets=None, data=model, min_responses=self.min_clients,
            )
            if not results or len(results) < self.min_clients:
                got = len(results) if results else 0
                logger.warning(
                    "Round %d: only %d/%d responses; skipping aggregation",
                    round_idx, got, self.min_clients,
                )
                continue

            aggregated = self._aggregate(results, model.params)
            model.params = aggregated
            model.current_round = round_idx + 1

            # ── Uplink bookkeeping + eval aggregation ──
            round_uplink = _sum_uplink_bytes(results)
            self.cumulative_uplink_bytes += round_uplink
            eval_summary = _aggregate_eval_metrics(results)

            logger.info(
                "Comm round %d: uplink=%s (Σ=%s) downlink=%s (Σ=%s)",
                round_idx + 1,
                _fmt_bytes(round_uplink),
                _fmt_bytes(self.cumulative_uplink_bytes),
                _fmt_bytes(round_downlink),
                _fmt_bytes(self.cumulative_downlink_bytes),
            )
            if eval_summary is not None:
                logger.info(
                    "Eval round %d: loss=%.4f ppl=%.2f (from %d clients, %d tokens)",
                    round_idx + 1,
                    eval_summary["eval_loss"],
                    eval_summary["perplexity"],
                    eval_summary["clients"],
                    eval_summary["tokens"],
                )

            # Surface the round's comm numbers on the model meta so any
            # downstream logger/persister (e.g. a JSONL writer) can pick them
            # up without re-summing from results.
            if model.meta is None:
                model.meta = {}
            model.meta["round_uplink_bytes"] = int(round_uplink)
            model.meta["round_downlink_bytes"] = int(round_downlink)
            model.meta["cumulative_uplink_bytes"] = int(self.cumulative_uplink_bytes)
            model.meta["cumulative_downlink_bytes"] = int(self.cumulative_downlink_bytes)
            if eval_summary is not None:
                model.meta["eval_loss"] = eval_summary["eval_loss"]
                model.meta["perplexity"] = eval_summary["perplexity"]

            # Update floor monitor from this round's profiles.
            if self.floor_monitor is not None and self.client_profiles:
                self.floor_monitor.update(self.client_profiles)
                logger.info(self.floor_monitor.stability_report())

            if (round_idx + 1) % 10 == 0:
                self.save_model(model)

        self.save_model(model)
        logger.info("Training complete.")

    # ── Aggregation ─────────────────────────────────────────────────────────
    def _aggregate(
        self,
        results: Dict[str, FLModel],
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        client_meta = self._collect_meta(results)

        # Classify every param present in *some* client's update.
        all_names: set[str] = set()
        for r in results.values():
            all_names.update(r.params.keys())

        by_role: Dict[str, list[str]] = {"expert": [], "router": [], "shared": []}
        for name in all_names:
            by_role[classify_param(name)].append(name)

        logger.info(
            "Aggregating: %d shared | %d router | %d expert",
            len(by_role["shared"]), len(by_role["router"]), len(by_role["expert"]),
        )

        out: Dict[str, np.ndarray] = {}

        # ── Shared params (attention, norms, embeddings) ──
        total_tokens = sum(max(m["total_tokens"], 1) for m in client_meta.values())
        for name in by_role["shared"]:
            weighted = self._weighted_sum(
                results, name,
                {c: max(client_meta[c]["total_tokens"], 1) / total_tokens for c in results},
            )
            if weighted is not None:
                out[name] = weighted
            elif name in global_params:
                out[name] = global_params[name]

        # ── Router params ──
        if self.cfg.use_router_alignment and self.client_profiles:
            router_weights, entropy = compute_router_weights(
                self.client_profiles, tau_active=self.cfg.tau_active,
            )
            logger.info("router_alignment entropy=%.3f weights=%s",
                        entropy, {k: round(v, 3) for k, v in router_weights.items()})
        else:
            uniform = 1.0 / max(len(results), 1)
            router_weights = {c: uniform for c in results}
            logger.info("router_alignment: disabled or unavailable; using uniform")

        for name in by_role["router"]:
            # Router weights are keyed by client name; some clients may not have
            # this param (if they froze it). Renormalise to contributors.
            contributor_names = [c for c in results if name in results[c].params]
            if not contributor_names:
                if name in global_params:
                    out[name] = global_params[name]
                continue
            sub = {c: router_weights.get(c, 0.0) for c in contributor_names}
            s = sum(sub.values()) or 1.0
            sub = {c: v / s for c, v in sub.items()}
            weighted = self._weighted_sum(results, name, sub)
            if weighted is not None:
                out[name] = weighted

        # ── Expert params ──
        floor_count = 0
        for name in by_role["expert"]:
            layer_idx, expert_idx = parse_expert_indices(name)
            contributions: list[tuple[str, np.ndarray, float]] = []
            for client_name, result in results.items():
                if name not in result.params:
                    continue
                if name in client_meta[client_name]["skipped_experts"]:
                    continue
                freq = 0.0
                af = client_meta[client_name]["activation_freq"]
                if af is not None and layer_idx is not None:
                    freq = float(af[layer_idx, expert_idx])
                contributions.append((client_name, np.asarray(result.params[name], dtype=np.float32), freq))

            if not contributions:
                if name in global_params:
                    out[name] = global_params[name]
                    floor_count += 1
                continue

            # NOTE: ??
            total_freq = sum(f for _, _, f in contributions)
            if total_freq < 1e-10:
                stacked = np.stack([p for _, p, _ in contributions], axis=0)
                out[name] = stacked.mean(axis=0)
            else:
                agg = np.zeros_like(contributions[0][1])
                for _c, p, f in contributions:
                    agg += (f / total_freq) * p
                out[name] = agg

        logger.info(
            "Expert aggregation: %d updated, %d kept from global (no contributors)",
            len(by_role["expert"]) - floor_count, floor_count,
        )
        return out

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _collect_meta(self, results: Dict[str, FLModel]) -> Dict[str, Dict[str, Any]]:
        per_client: Dict[str, Dict[str, Any]] = {}
        self.client_profiles = {}
        for client_name, r in results.items():
            meta = r.meta or {}
            act = meta.get("activation_profile", {}) or {}
            freq = act.get("activation_freq")
            tokens = act.get("total_tokens", 1) or 1
            if freq is not None:
                freq_arr = np.asarray(freq, dtype=np.float64)
                self.client_profiles[client_name] = freq_arr
            else:
                freq_arr = None
            per_client[client_name] = {
                "activation_freq": freq_arr,
                "total_tokens": int(tokens),
                "skipped_experts": set(meta.get("skipped_experts", [])),
            }
        return per_client

    @staticmethod
    def _weighted_sum(
        results: Dict[str, FLModel],
        name: str,
        weights: Dict[str, float],
    ) -> Optional[np.ndarray]:
        acc: Optional[np.ndarray] = None
        for client_name, result in results.items():
            if name not in result.params:
                continue
            w = weights.get(client_name, 0.0)
            if w == 0.0:
                continue
            arr = np.asarray(result.params[name], dtype=np.float32)
            acc = w * arr if acc is None else acc + w * arr
        return acc


# ── Module-level comm & eval helpers (pure-numpy, test-friendly) ───────────

def _estimate_params_bytes(params: Optional[Dict[str, np.ndarray]]) -> int:
    """Sum of ``nbytes`` over a params dict — approximates on-the-wire size.

    We intentionally ignore NVFlare framing overhead: the metric we want to
    report is "how many bytes of model payload did we move", not wire bytes.
    """
    if not params:
        return 0
    total = 0
    for v in params.values():
        if hasattr(v, "nbytes"):
            total += int(v.nbytes)
        else:
            # INT8-packed experts arrive as {"quantized": np.ndarray, "scale": float}.
            if isinstance(v, dict) and "quantized" in v:
                q = v["quantized"]
                total += int(getattr(q, "nbytes", 0)) + 4  # scale is fp32
            else:
                arr = np.asarray(v)
                total += int(arr.nbytes)
    return total


def _sum_uplink_bytes(results: Dict[str, FLModel]) -> int:
    """Sum client-reported ``bytes_compressed`` + shared params.

    Expert deltas are carried through the SparseMoEEncoder filter which
    records ``bytes_compressed`` in meta. Shared params (attention, norms,
    embeddings) pass through uncompressed, so their size is whatever the
    params dict carries. We add both so the number reflects total uplink,
    not just the compressed-expert portion.
    """
    total = 0
    for r in results.values():
        meta = r.meta or {}
        comp_meta = meta.get("compression_metadata")
        if comp_meta is not None:
            total += int(comp_meta.get("bytes_compressed", 0))
            # Params dict still holds shared/non-expert entries as ndarrays.
            # Subtract the expert entries' sizes from the full params-bytes to
            # avoid double-counting — but that needs the compressor to have
            # tagged expert names, which it has via ``compression_map``.
            expert_names = set(comp_meta.get("compression_map", {}).keys())
            for name, v in (r.params or {}).items():
                if name in expert_names:
                    continue
                if hasattr(v, "nbytes"):
                    total += int(v.nbytes)
                else:
                    total += int(np.asarray(v).nbytes)
        else:
            # No compressor in the pipeline: count the raw params dict.
            total += _estimate_params_bytes(r.params)
    return total


def _aggregate_eval_metrics(results: Dict[str, FLModel]) -> Optional[Dict[str, float]]:
    """Token-weighted mean of client eval losses → server-side perplexity.

    Returns ``None`` if no client reported eval stats (e.g. EVAL_FRAC=0).
    Weighting by eval tokens (not sample count) keeps this comparable with the
    client-side computation, which is itself token-weighted.
    """
    loss_sum, token_sum, n_clients = 0.0, 0, 0
    for r in results.values():
        meta = r.meta or {}
        if "eval_loss" not in meta:
            continue
        tokens = int(meta.get("eval_tokens", 0))
        if tokens <= 0:
            continue
        loss_sum += float(meta["eval_loss"]) * tokens
        token_sum += tokens
        n_clients += 1
    if token_sum == 0:
        return None
    avg = loss_sum / token_sum
    return {
        "eval_loss": avg,
        "perplexity": float(np.exp(min(avg, 50.0))),
        "tokens": token_sum,
        "clients": n_clients,
    }


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
