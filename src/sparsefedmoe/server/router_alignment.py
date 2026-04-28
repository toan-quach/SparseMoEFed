"""Alignment-Weighted Router Aggregation (arch §3.4).

Given the ``{client: activation_freq}`` map already collected for freq-weighted
expert aggregation, compute a normalized alignment weight per client. The
weights replace uniform / dataset-size weighting when averaging router (gate)
parameters so that clients whose active experts overlap with the federation's
global distribution have proportionally more influence on routing decisions.

Four steps from §3.4:
    1. Active set:      S_c = {e : freq_c(e) > tau_active}
    2. Popularity:      popularity(e) = |{c : e ∈ S_c}| / N
    3. Alignment:       alignment(c) = Σ_{e ∈ S_c} popularity(e)
    4. Normalize:       weight(c) = alignment(c) / Σ alignment

Edge cases:
  - If a client has an empty active set, its alignment is 0; we fall back to
    uniform weighting rather than dropping the client entirely.
  - If *all* alignments are zero (degenerate, but possible on the first round
    if tau_active is mis-set), fall back to uniform weighting across all
    clients.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def compute_router_weights(
    client_profiles: Dict[str, np.ndarray],
    tau_active: float = 0.01,
) -> Tuple[Dict[str, float], float]:
    """Return ``({client: weight}, entropy)``.

    ``entropy`` is the Shannon entropy of the normalised weight distribution
    (in nats); it's a quick diagnostic that the aggregator logs so operators
    can see whether alignment is concentrating weight on a few clients.
    """
    names = sorted(client_profiles.keys())
    n = len(names)
    if n == 0:
        return {}, 0.0
    if n == 1:
        return {names[0]: 1.0}, 0.0

    # Build the active-set masks in one vectorised shot.
    stack = np.stack([np.asarray(client_profiles[name]) for name in names], axis=0)
    # Shape: (N, L, E), bool.
    active = stack > tau_active

    # popularity(e) per (layer, expert): mean across clients of the active mask.
    popularity = active.mean(axis=0)  # shape (L, E)

    # alignment(c) = sum over (l, e) in S_c of popularity(l, e).
    alignment = (active.astype(np.float64) * popularity[None, :, :]).sum(axis=(1, 2))
    total = alignment.sum()

    if total <= 0.0:
        uniform = 1.0 / n
        weights = {name: uniform for name in names}
        # Entropy of the uniform distribution.
        return weights, math.log(n)

    norm = alignment / total
    # Fallback: clients with zero alignment get a small uniform floor so router
    # params from a maverick client are not entirely dropped. We use 1/(10*N) of
    # the total weight, renormalised.
    floor = (1.0 / (10 * n)) if (alignment == 0).any() else 0.0
    weights_arr = norm + floor
    weights_arr = weights_arr / weights_arr.sum()

    entropy = float(-np.sum(weights_arr * np.log(np.clip(weights_arr, 1e-12, None))))
    return {name: float(w) for name, w in zip(names, weights_arr)}, entropy
