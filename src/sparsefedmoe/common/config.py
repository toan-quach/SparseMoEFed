"""Single source of truth for SparseFedMoE thresholds and feature flags.

Both the client-side encoder filter and the server-side controller are
instantiated from the same dataclass (serialized through NVFlare job configs),
so the whole stack agrees on what counts as HIGH / MEDIUM / SKIP and which
optional components are active.

Defaults match the architecture doc §3.2 (``τ_skip = 0.005``, ``τ_high = 0.05``)
and §3.5 (``γ = 0.10``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass
class SparseFedMoEConfig:
    # ── Per-expert compression (§3.2) ──
    tau_skip: float = 0.005
    tau_high: float = 0.05
    use_error_feedback: bool = True
    ef_flush_threshold: float = 0.0  # 0 ⇒ only flush when promoted by floor

    # ── Global floor monitor (§3.5) ──
    use_floor: bool = True
    floor_gamma: float = 0.10
    floor_min_clients: int = 2  # don't compute floor with < N client reports

    # ── Frequency-weighted expert aggregation (§3.3) ──
    use_freq_weighted_experts: bool = True

    # ── Alignment-weighted router aggregation (§3.4) ──
    use_router_alignment: bool = True
    tau_active: float = 0.01  # expert is "active" on a client if freq ≥ this

    # ── Expert-affinity clustering (§3.6; optional add-on) ──
    use_clustering: bool = False
    num_clusters: int = 0  # 0 ⇒ use similarity_threshold
    cluster_similarity_threshold: float = 0.85
    cluster_recluster_every: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SparseFedMoEConfig":
        # Ignore unknown keys so older job configs keep working.
        fields = {k: data[k] for k in data if k in cls.__dataclass_fields__}
        return cls(**fields)
