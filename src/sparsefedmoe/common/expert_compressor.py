"""Adaptive per-expert compression with EF21-style error feedback.

Migrated from sparsefedmoe_tmp. Behavior preserved; parse-helper is shared
with the rest of the package via ``common.model_utils``.

Tier rule (arch §3.2):

    freq >= high_threshold          -> FP16
    skip_threshold <= freq < high   -> INT8 symmetric quantization
    freq < skip_threshold           -> SKIP (accumulate residual; flush on norm trigger)

Error-feedback contract: whatever the tier decides to *transmit*, the residual
between the true delta and the transmitted approximation goes into a per-(layer,
expert, param) buffer. On SKIP that residual IS the full delta. The buffer is
added to next round's delta before re-classifying, and flushed back out once the
accumulated norm exceeds ``ef_flush_threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .model_utils import parse_expert_indices


@dataclass
class CompressionConfig:
    skip_threshold: float = 0.005
    high_threshold: float = 0.05
    use_error_feedback: bool = True
    # L2 norm threshold at which a SKIP residual buffer is flushed as an INT8 correction.
    # Defaults to 0.0 == never flush proactively; callers should set this explicitly.
    ef_flush_threshold: float = 0.0


class ExpertCompressor:
    """Stateful compressor; EF buffers persist across rounds on the client."""

    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        # Key: (layer_idx, expert_idx, param_name) -> residual tensor on CPU.
        self.error_buffers: Dict[Tuple[int, int, str], torch.Tensor] = {}

    # ── Public API ──────────────────────────────────────────────────────────
    def compress_expert_updates(
        self,
        expert_deltas: Dict[str, torch.Tensor],
        activation_freq: np.ndarray,
        floor_tiers: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Compress a dict of per-expert deltas.

        Args:
            expert_deltas: ``{param_name: delta_tensor}``. Non-expert names are
                pass-through (tier "none").
            activation_freq: shape ``(num_layers, num_experts)``.
            floor_tiers: optional ``{(l,e): "INT8"}`` from the Global Floor Monitor.
                An expert listed here is promoted to at least INT8 even when its
                local frequency is below ``skip_threshold``.
        """
        compressed: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {
            "compression_map": {},
            "original_dtypes": {},
            "original_shapes": {},
            "bytes_original": 0,
            "bytes_compressed": 0,
            "ef_flushes": [],
        }
        floor_tiers = floor_tiers or {}

        for name, delta in expert_deltas.items():
            layer_idx, expert_idx = parse_expert_indices(name)
            original_bytes = delta.nelement() * delta.element_size()
            metadata["bytes_original"] += original_bytes

            if layer_idx is None:
                # Not an expert param: pass-through at whatever dtype it arrives in.
                compressed[name] = delta
                metadata["compression_map"][name] = "none"
                metadata["bytes_compressed"] += original_bytes
                continue

            metadata["original_dtypes"][name] = str(delta.dtype)
            metadata["original_shapes"][name] = list(delta.shape)

            # Fold any previously-stashed residual into this round's delta before
            # tier selection (EF21 step: x_t <- x_t + e_{t-1}).
            ef_key = (layer_idx, expert_idx, name)
            if self.config.use_error_feedback and ef_key in self.error_buffers:
                delta = delta + self.error_buffers[ef_key].to(delta.device)

            freq = float(activation_freq[layer_idx, expert_idx])
            tier = self._assign_tier(freq)

            # Server-mandated floor promotion (arch §3.5).
            floor = floor_tiers.get((layer_idx, expert_idx))
            if floor is not None and _tier_rank(floor) > _tier_rank(tier):
                tier = floor

            if tier == "SKIP":
                compressed[name] = None
                metadata["compression_map"][name] = "skipped"
                metadata["bytes_compressed"] += 0
                if self.config.use_error_feedback:
                    self.error_buffers[ef_key] = delta.detach().cpu()
                    # Proactive flush if the residual's norm has crossed threshold.
                    if self.config.ef_flush_threshold > 0.0:
                        norm = float(self.error_buffers[ef_key].norm())
                        if norm > self.config.ef_flush_threshold:
                            flushed = self._flush_residual_as_int8(ef_key, metadata)
                            compressed[name] = flushed
                            metadata["compression_map"][name] = "int8"

            elif tier == "FP16":
                out = delta.to(torch.float16)
                compressed[name] = out
                metadata["compression_map"][name] = "fp16"
                metadata["bytes_compressed"] += out.nelement() * 2
                if self.config.use_error_feedback:
                    self.error_buffers.pop(ef_key, None)

            elif tier == "INT8":
                packed = self._quantize_int8(delta)
                compressed[name] = packed
                metadata["compression_map"][name] = "int8"
                metadata["bytes_compressed"] += delta.nelement() + 4  # + fp32 scale
                if self.config.use_error_feedback:
                    reconstructed = self._dequantize_int8(packed, delta.shape)
                    residual = delta - reconstructed.to(delta.device)
                    self.error_buffers[ef_key] = residual.detach().cpu()

            else:
                raise AssertionError(f"unknown tier {tier!r}")

        metadata["compression_ratio"] = (
            metadata["bytes_compressed"] / metadata["bytes_original"]
            if metadata["bytes_original"] > 0
            else 1.0
        )
        return compressed, metadata

    def decompress_expert_updates(
        self,
        compressed: Dict[str, Any],
        metadata: Dict,
    ) -> Dict[str, torch.Tensor]:
        """Inverse of ``compress_expert_updates`` (server-side)."""
        out: Dict[str, torch.Tensor] = {}
        for name, data in compressed.items():
            tier = metadata["compression_map"].get(name, "none")
            if tier == "skipped" or data is None:
                continue
            if tier == "none":
                out[name] = data
            elif tier == "fp16":
                out[name] = data.to(torch.float32) if hasattr(data, "to") else torch.as_tensor(data, dtype=torch.float32)
            elif tier == "int8":
                shape = metadata["original_shapes"][name]
                out[name] = self._dequantize_int8(data, shape)
        return out

    # ── Helpers ─────────────────────────────────────────────────────────────
    def _assign_tier(self, freq: float) -> str:
        if freq < self.config.skip_threshold:
            return "SKIP"
        if freq >= self.config.high_threshold:
            return "FP16"
        return "INT8"

    def _flush_residual_as_int8(
        self,
        ef_key: Tuple[int, int, str],
        metadata: Dict,
    ) -> Dict:
        residual = self.error_buffers.pop(ef_key)
        packed = self._quantize_int8(residual)
        metadata["bytes_compressed"] += residual.nelement() + 4
        metadata["ef_flushes"].append(ef_key[2])
        metadata["original_shapes"][ef_key[2]] = list(residual.shape)
        return packed

    @staticmethod
    def _quantize_int8(tensor: torch.Tensor) -> Dict[str, Any]:
        # The problem: Symmetric INT8 quantization maps [-max_val, +max_val] to [-127, +127]. 
        # The scale factor is max_val / 127. If the tensor is near-zero (e.g., an expert that 
        # barely trained this round, or an EF21 residual that's nearly converged), max_val ≈ 0 
        # and scale ≈ 0.
        # 
        # Without the fix: Division by scale in the quantization step (flat / scale) produces 
        # inf or nan. Those corrupted INT8 values travel to the server, get dequantized as garbage, 
        # and pollute the global expert weights. One near-zero expert delta could corrupt the entire 
        # expert.
        # 
        # Why 1e-10: It's effectively saying "if the delta is smaller than 1e-10 in magnitude, 
        # treat it as if the largest value is 1e-10." The quantized result will be all zeros 
        # (since tiny_value / 1e-10 rounds to 0 in int8), which is the correct semantic, the 
        # delta was negligible, so transmit nothing. The torch.clamp(..., -127, 127) on the next line 
        # is a belt-and-suspenders guard ensuring the INT8 range is respected even if floating-point 
        # arithmetic produces values slightly outside bounds.

        flat = tensor.detach().float().flatten()
        scale = flat.abs().max() / 127.0
        if scale < 1e-10:
            scale = torch.tensor(1e-10)
        quantized = torch.clamp(torch.round(flat / scale), -127, 127).to(torch.int8)
        return {"quantized": quantized, "scale": float(scale)}

    @staticmethod
    def _dequantize_int8(data: Dict, shape) -> torch.Tensor:
        q = data["quantized"]
        if isinstance(q, np.ndarray):
            q = torch.from_numpy(q)
        return (q.float() * data["scale"]).reshape(shape)

    # Legacy parse helper kept for external callers that import from here.
    @staticmethod
    def _parse_expert_param(name: str) -> Tuple[Optional[int], Optional[int]]:
        return parse_expert_indices(name)

    def get_compression_stats(self, metadata: Dict) -> str:
        cm = metadata.get("compression_map", {})
        counts = {"skipped": 0, "fp16": 0, "int8": 0, "none": 0}
        for v in cm.values():
            counts[v] = counts.get(v, 0) + 1
        orig = metadata.get("bytes_original", 0)
        comp = metadata.get("bytes_compressed", 0)
        ratio = metadata.get("compression_ratio", 1.0)
        savings = (1 - ratio) * 100
        return (
            f"Compression Stats:\n"
            f"  Skipped: {counts['skipped']} | INT8: {counts['int8']} | "
            f"FP16: {counts['fp16']} | Pass-through: {counts['none']}\n"
            f"  Original: {orig / 1e6:.1f} MB | Compressed: {comp / 1e6:.1f} MB | "
            f"Savings: {savings:.1f}%"
        )


_TIER_ORDER = {"SKIP": 0, "INT8": 1, "FP16": 2}


def _tier_rank(tier: str) -> int:
    return _TIER_ORDER.get(tier.upper(), 0)
