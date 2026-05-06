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

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

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
        # Clients build their own model and seed the first round's params, so
        # the persistor's initial state is empty by design — opt in to that.
        super().__init__(allow_empty_global_weights=True)
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
        self.round_metrics: List[Dict[str, Any]] = []

    # ── Main control flow ───────────────────────────────────────────────────
    def run(self):
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

            result_list = self.send_model_and_wait(
                targets=None, data=model, min_responses=self.min_clients,
            )
            if not result_list or len(result_list) < self.min_clients:
                got = len(result_list) if result_list else 0
                logger.warning(
                    "Round %d: only %d/%d responses; skipping aggregation",
                    round_idx, got, self.min_clients,
                )
                continue

            # Re-key by client name so the aggregation code below — which was
            # written when send_model_and_wait returned Dict[str, FLModel] —
            # keeps working under 2.6's List[FLModel] return contract.
            results: Dict[str, FLModel] = {}
            for i, r in enumerate(result_list):
                cname = (r.meta or {}).get("client_name") or f"client_{i}"
                results[cname] = r

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

            # ── Record round metrics ──
            round_record: Dict[str, Any] = {
                "round": round_idx + 1,
                "num_clients": len(results),
                "round_uplink_bytes": int(round_uplink),
                "round_downlink_bytes": int(round_downlink),
                "cumulative_uplink_bytes": int(self.cumulative_uplink_bytes),
                "cumulative_downlink_bytes": int(self.cumulative_downlink_bytes),
            }
            if eval_summary is not None:
                round_record["eval_loss"] = eval_summary["eval_loss"]
                round_record["perplexity"] = eval_summary["perplexity"]
                round_record["eval_tokens"] = eval_summary["tokens"]
                round_record["eval_clients"] = eval_summary["clients"]
            if self.floor_monitor is not None:
                fl = self.floor_monitor.get_floor_tier_list()
                round_record["floor_protected_experts"] = len(fl) if fl else 0
            self.round_metrics.append(round_record)

            # Update floor monitor from this round's profiles.
            if self.floor_monitor is not None and self.client_profiles:
                self.floor_monitor.update(self.client_profiles)
                logger.info(self.floor_monitor.stability_report())

            if (round_idx + 1) % 10 == 0:
                self.save_model(model)

        self.save_model(model)
        self._save_metrics_report()
        logger.info("Training complete.")

    # ── Aggregation ─────────────────────────────────────────────────────────
    def _aggregate(
        self,
        results: Dict[str, FLModel],
        global_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        client_meta = self._collect_meta(results)

        # Classify every param present in *some* client's update OR in the
        # current global model.  Without the global-param union, an expert
        # skipped by every client in a round silently disappears from the
        # global model instead of being preserved unchanged.
        # TODO: revisit whether universally-skipped experts should decay or
        #       be pruned rather than preserved indefinitely.
        all_names: set[str] = set()
        for r in results.values():
            all_names.update(r.params.keys())
        if global_params:
            all_names.update(global_params.keys())

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
            s = sum(sub.values()) or 1.0    # Avoid division by zero if all contributors have zero router weight.
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
                if self.cfg.use_freq_weighted_experts:
                    af = client_meta[client_name]["activation_freq"]
                    if af is not None and layer_idx is not None:
                        freq = float(af[layer_idx, expert_idx])
                contributions.append((client_name, np.asarray(result.params[name], dtype=np.float32), freq))

            if not contributions:
                if name in global_params:
                    out[name] = global_params[name]
                    floor_count += 1
                continue

            if not self.cfg.use_freq_weighted_experts:
                contrib_tokens = sum(
                    max(client_meta[c]["total_tokens"], 1) for c, _, _ in contributions
                )
                agg = np.zeros_like(contributions[0][1])
                for c, p, _ in contributions:
                    agg += (max(client_meta[c]["total_tokens"], 1) / contrib_tokens) * p
                out[name] = agg
            else:
                total_freq = sum(f for _, _, f in contributions)
                if total_freq < 1e-10:
                    # The problem: Expert aggregation (section 3.3) weights each client's 
                    # contribution by how frequently that client activated the expert. If 
                    # an expert was activated by multiple clients but at vanishingly small 
                    # frequencies (all near zero - possible in early rounds before routing 
                    # stabilizes), total_freq ≈ 0 and dividing f / total_freq would produce 
                    # inf weights.
                    # 
                    # Without the fix: One client whose frequency is 1e-15 while others are 
                    # 1e-16 would get essentially 100% of the weight — an arbitrary winner 
                    # determined by floating-point noise. The aggregated expert weights would 
                    # be just one client's update, not a meaningful combination.
                    # 
                    # The fallback: Unweighted mean. When no client has a meaningful frequency 
                    # signal for this expert, all contributions are treated equally. This is 
                    # the honest answer: "we don't have enough activation data to rank these 
                    # contributions, so average them uniformly." The 1e-10 threshold is generous,
                    # any total frequency above it reflects real activation signal.
                    # 
                    # Contrast with decision #1: Decision #1 injects a floor to keep a client alive; 
                    # this one switches aggregation strategy entirely when the weighting signal is too weak.

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

    def _save_metrics_report(self):
        if not self.round_metrics:
            return

        completed = len(self.round_metrics)
        total_uplink = self.cumulative_uplink_bytes
        total_downlink = self.cumulative_downlink_bytes
        total_comm = total_uplink + total_downlink

        summary: Dict[str, Any] = {
            "rounds_completed": completed,
            "total_uplink_bytes": total_uplink,
            "total_downlink_bytes": total_downlink,
            "total_comm_bytes": total_comm,
            "avg_uplink_per_round": total_uplink / max(completed, 1),
            "avg_downlink_per_round": total_downlink / max(completed, 1),
        }

        eval_rounds = [r for r in self.round_metrics if "eval_loss" in r]
        if eval_rounds:
            best = min(eval_rounds, key=lambda r: r["eval_loss"])
            final = eval_rounds[-1]
            summary["best_eval_loss"] = best["eval_loss"]
            summary["best_perplexity"] = best["perplexity"]
            summary["best_round"] = best["round"]
            summary["final_eval_loss"] = final["eval_loss"]
            summary["final_perplexity"] = final["perplexity"]

        report = {"summary": summary, "rounds": self.round_metrics}

        logger.info(
            "══ Training Summary ══\n"
            "  Rounds completed:   %d\n"
            "  Total uplink:       %s\n"
            "  Total downlink:     %s\n"
            "  Total comm:         %s\n"
            "  Avg uplink/round:   %s\n"
            "  Avg downlink/round: %s",
            completed,
            _fmt_bytes(total_uplink),
            _fmt_bytes(total_downlink),
            _fmt_bytes(total_comm),
            _fmt_bytes(int(summary["avg_uplink_per_round"])),
            _fmt_bytes(int(summary["avg_downlink_per_round"])),
        )
        if eval_rounds:
            logger.info(
                "  Best eval loss:     %.4f (ppl=%.2f, round %d)\n"
                "  Final eval loss:    %.4f (ppl=%.2f)",
                summary["best_eval_loss"], summary["best_perplexity"],
                summary["best_round"],
                summary["final_eval_loss"], summary["final_perplexity"],
            )

        try:
            workspace = self.engine.get_workspace()
            run_dir = workspace.get_run_dir(self.fl_ctx.get_job_id())
            metrics_path = os.path.join(run_dir, "metrics.json")
        except Exception:
            metrics_path = "metrics.json"
        try:
            with open(metrics_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("Metrics saved to %s", metrics_path)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not write metrics.json: %s", e)

    @staticmethod
    def _weighted_sum(
        results: Dict[str, FLModel],
        name: str,
        weights: Dict[str, float],
    ) -> Optional[np.ndarray]:
        contributors = {
            c: np.asarray(r.params[name], dtype=np.float32)
            for c, r in results.items()
            if name in r.params and weights.get(c, 0.0) != 0.0
        }
        if not contributors:
            return None
        w_sum = sum(weights[c] for c in contributors)
        if w_sum <= 0.0:
            return None
        acc: Optional[np.ndarray] = None
        for c, arr in contributors.items():
            w = weights[c] / w_sum
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
