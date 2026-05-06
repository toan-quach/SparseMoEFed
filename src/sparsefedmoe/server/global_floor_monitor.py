"""Global Floor Monitor (arch §3.5).

Aggregates per-expert activation frequencies across all clients reporting in a
round, identifies experts whose global frequency falls below ``γ × mean_freq``
(computed per-layer to avoid a layer-depth bias), and publishes a set of
floor-protected ``(layer, expert)`` pairs. Downstream: ``FreqWeightedFedAvg``
attaches these tiers to the next round's broadcast so the client encoder
promotes them from SKIP to at least INT8.

The monitor keeps a short running history of floor sets so we can log how
stable the set is round-to-round — useful signal for debugging expert collapse.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np


Tier = str  # "INT8" in the common case
ExpertKey = Tuple[int, int]


class GlobalFloorMonitor:
    def __init__(
        self,
        floor_gamma: float = 0.10,
        min_clients: int = 2,
        history_size: int = 5,
        default_tier: Tier = "INT8",
    ):
        self.floor_gamma = float(floor_gamma)
        self.min_clients = int(min_clients)
        self.default_tier = default_tier
        self._history: Deque[Set[ExpertKey]] = deque(maxlen=history_size)
        self._last: Dict[ExpertKey, Tier] = {}

    def update(
        self,
        client_profiles: Dict[str, np.ndarray],
    ) -> Dict[ExpertKey, Tier]:
        """Recompute the floor set from this round's client profiles.

        ``client_profiles`` values are ``(num_layers, num_experts)`` matrices
        whose per-layer rows sum to ``top_k`` (the invariant from §3.1).
        """
        if len(client_profiles) < self.min_clients:
            # Not enough signal — keep the prior floor set so protection persists.
            # The problem: The Global Floor Monitor (section 3.5) identifies under-utilized 
            # experts and protects them by promoting their compression tier from SKIP to INT8. 
            # It needs a representative sample of client activation profiles to compute global 
            # frequency. If only 1 client reports in a round (maybe others timed out or 
            # disconnected), the "global" frequency is really just one client's local frequency
            # highly noisy.
            # 
            # Without the fix: The floor set would swing wildly round-to-round based on whichever 
            # subset of clients happened to respond. An expert that genuinely needs protection 
            # could lose it because the one client that responded happened to use different experts. 
            # Then that expert gets SKIPped by all clients next round, its weights stagnate, and it 
            # may never recover — this is the "expert collapse" problem the floor monitor exists to 
            # prevent.
            # 
            # The design: Return the previous floor set unchanged. Protection persists even during 
            # low-turnout rounds. The min_clients default of 2 means you need at least two independent 
            # views before the monitor updates its beliefs. The _history deque (used in stability_report()) 
            # lets operators see churn — if the floor set is changing every round, something is wrong.
            
            return dict(self._last)

        stack = np.stack([np.asarray(p) for p in client_profiles.values()], axis=0)
        # Sum across clients → per-(layer, expert) global frequency.
        global_freq = stack.sum(axis=0)  # shape (L, E)

        floor: Dict[ExpertKey, Tier] = {}
        # Per-layer thresholding: an expert is protected if it's underutilised
        # *within its own layer*, not relative to the whole model.
        mean_per_layer = global_freq.mean(axis=1, keepdims=True)  # (L, 1)
        threshold = self.floor_gamma * mean_per_layer            # (L, 1)
        mask = global_freq < threshold                           # (L, E) bool

        for l, e in zip(*np.where(mask)):
            floor[(int(l), int(e))] = self.default_tier

        self._last = floor
        self._history.append(set(floor.keys()))
        return floor

    def get_floor_tiers(self) -> Dict[ExpertKey, Tier]:
        return dict(self._last)

    def get_floor_tier_list(self) -> List[List]:
        """Serialization-friendly form: ``[[l, e, tier], …]``."""
        return [[l, e, t] for (l, e), t in sorted(self._last.items())]

    def stability_report(self) -> str:
        if len(self._history) < 2:
            return f"floor_protected_count={len(self._last)}"
        latest = self._history[-1]
        prev = self._history[-2]
        churn = len(latest.symmetric_difference(prev))
        return (
            f"floor_protected_count={len(latest)} "
            f"churn_vs_prev={churn} "
            f"history_len={len(self._history)}"
        )
