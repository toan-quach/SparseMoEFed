"""SparseMoEEncoder — NVFlare Filter for client-side sparse expert encoding (arch §3.2).

Reads:
  - ``activation_profile`` from the outgoing DXO metadata (set by the trainer via
    ``ActivationTracker.get_activation_metadata``).
  - ``floor_tiers`` from the *incoming* global model metadata (set by the server's
    Global Floor Monitor). We persist the last-seen floor tiers as Filter state
    so we can apply them even though NVFlare's Filter.process() only sees the
    outgoing task result.

Writes:
  - Compressed per-expert deltas (FP16 / INT8 dicts / omitted for SKIP).
  - ``compression_metadata`` describing each param's tier and original shape, for
    the server-side decoder.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.filter import Filter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

from sparsefedmoe.common.config import SparseFedMoEConfig
from sparsefedmoe.common.expert_compressor import CompressionConfig, ExpertCompressor
from sparsefedmoe.common.model_utils import parse_expert_indices

logger = logging.getLogger(__name__)


# NVFlare FL context key under which we stash the last-seen floor tiers.
_FLOOR_TIERS_CTX_KEY = "sparsefedmoe.floor_tiers"


class SparseMoEEncoder(Filter):
    """Client filter: compress outgoing MoE expert updates by activation tier."""

    def __init__(
        self,
        tau_skip: float = 0.005,
        tau_high: float = 0.05,
        use_error_feedback: bool = True,
        ef_flush_threshold: float = 0.0,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # Prefer a fully-specified ``config`` dict (set by job config) over the
        # individual kwargs, but either works.
        if config is not None:
            cfg = SparseFedMoEConfig.from_dict(config)
        else:
            cfg = SparseFedMoEConfig(
                tau_skip=tau_skip,
                tau_high=tau_high,
                use_error_feedback=use_error_feedback,
                ef_flush_threshold=ef_flush_threshold,
            )
        self.cfg = cfg

        self.compressor = ExpertCompressor(
            CompressionConfig(
                skip_threshold=cfg.tau_skip,
                high_threshold=cfg.tau_high,
                use_error_feedback=cfg.use_error_feedback,
                ef_flush_threshold=cfg.ef_flush_threshold,
            )
        )

        # Floor tiers seen on the most recent broadcast. Applied next upload.
        self._floor_tiers: Dict[Tuple[int, int], str] = {}

        logger.info(
            "SparseMoEEncoder: tau_skip=%s tau_high=%s use_ef=%s",
            cfg.tau_skip, cfg.tau_high, cfg.use_error_feedback,
        )

    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Union[Shareable, None]:
        try:
            dxo = DXO.from_shareable(shareable)
        except Exception as e:  # noqa: BLE001 - NVFlare swallows errors silently otherwise
            logger.warning("SparseMoEEncoder: cannot parse DXO (%s); passing through", e)
            return shareable

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS):
            return shareable

        meta = dxo.get_meta_props() or {}
        activation_data = meta.get("activation_profile")
        if activation_data is None:
            logger.warning(
                "SparseMoEEncoder: no activation_profile in metadata; passing through. "
                "Ensure your trainer attaches ActivationTracker.get_activation_metadata()."
            )
            return shareable
        activation_freq = np.asarray(activation_data["activation_freq"], dtype=np.float64)

        # Pull floor tiers either from FL context (set by a companion task data filter
        # if present) or from our stashed copy.
        floor_from_ctx = fl_ctx.get_prop(_FLOOR_TIERS_CTX_KEY, None)
        floor_tiers = floor_from_ctx or self._floor_tiers

        # ── Split expert vs shared params ──
        params = dxo.data
        expert_params: Dict[str, Any] = {}
        shared_params: Dict[str, Any] = {}
        for name, value in params.items():
            l, e = parse_expert_indices(name)
            if l is not None:
                expert_params[name] = value
            else:
                shared_params[name] = value

        # Compressor expects torch tensors.
        expert_tensors = {
            k: (torch.from_numpy(v) if isinstance(v, np.ndarray) else v)
            for k, v in expert_params.items()
        }

        compressed, comp_metadata = self.compressor.compress_expert_updates(
            expert_tensors, activation_freq, floor_tiers=floor_tiers,
        )
        logger.info(self.compressor.get_compression_stats(comp_metadata))

        # ── Repack ──
        new_params: Dict[str, Any] = dict(shared_params)  # pass-through
        for name, value in compressed.items():
            if value is None:
                continue  # SKIP: omit entirely
            if isinstance(value, dict) and "quantized" in value:  # INT8 packed
                q = value["quantized"]
                q_np = q.numpy() if hasattr(q, "numpy") else np.asarray(q)
                new_params[name] = {"quantized": q_np, "scale": value["scale"]}
            elif hasattr(value, "numpy"):
                new_params[name] = value.numpy()
            else:
                new_params[name] = value

        new_meta = dict(meta)
        new_meta["compression_metadata"] = comp_metadata
        new_meta["skipped_experts"] = [
            name for name, t in comp_metadata["compression_map"].items() if t == "skipped"
        ]
        # Surface flushes in the metadata too (useful for tests and logging).
        new_meta["ef_flushes"] = comp_metadata.get("ef_flushes", [])

        return DXO(data_kind=dxo.data_kind, data=new_params, meta=new_meta).to_shareable()

    # Hook for a future task-data filter or controller-set property: update the
    # floor tiers state from a serializable representation (list of [l,e,tier]).
    def update_floor_tiers(self, floor_entries):
        self._floor_tiers = {
            (int(l), int(e)): str(tier) for (l, e, tier) in (floor_entries or [])
        }
