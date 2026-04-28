"""Per-expert activation tracker (arch §3.1).

Forward-hooks every ``SparseMoeBlock`` to count top-k expert selections per
layer. Exposes the normalised frequency matrix ``A ∈ ℝ^{L×E}`` (sums to k per
layer) as both a numpy array and a JSON-serializable metadata dict.

The hook costs one ``torch.bincount`` per layer per forward pass.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class ActivationTracker:
    def __init__(self, model, top_k: Optional[int] = None):
        self.model = model
        self.hooks: List[Any] = []
        self.tracking = False

        config = model.config
        self.num_layers: int = config.num_hidden_layers
        self.num_experts: int = config.num_experts
        self.top_k: int = top_k or config.num_experts_per_tok

        self.expert_counts = np.zeros(
            (self.num_layers, self.num_experts), dtype=np.int64
        )
        # Token count is per-layer so that the ``Σ_e freq[l][e] = k`` invariant
        # from arch §3.1 holds (each layer's hook sees its own token count).
        self.layer_tokens = np.zeros(self.num_layers, dtype=np.int64)

    # ── Lifecycle ──
    def start(self):
        self.reset()
        moe_layers = self._find_moe_layers()
        if not moe_layers:
            raise RuntimeError(
                "No MoE layers found: no module class name contained 'SparseMoe' "
                "or 'MoeBlock'. Is this model actually MoE?"
            )
        for layer_idx, (_name, module) in enumerate(moe_layers):
            self.hooks.append(module.register_forward_hook(self._make_hook(layer_idx)))
        self.tracking = True
        return self

    def stop(self):
        self.tracking = False
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def reset(self):
        self.expert_counts = np.zeros(
            (self.num_layers, self.num_experts), dtype=np.int64
        )
        self.layer_tokens = np.zeros(self.num_layers, dtype=np.int64)
        self.tracking = False

    @property
    def total_tokens(self) -> int:
        """Aggregate token count across layers (for reporting only)."""
        return int(self.layer_tokens.max())

    # ── Internals ──
    def _find_moe_layers(self) -> List[Tuple[str, torch.nn.Module]]:
        out = []
        for name, module in self.model.named_modules():
            cls = type(module).__name__
            if "SparseMoe" in cls or "MoeBlock" in cls:
                out.append((name, module))
        return out

    def _make_hook(self, layer_idx: int):
        def hook_fn(_module, _inputs, output):
            # After running, say, 1000 tokens through the model:

            # layer_tokens[l] = 1000
            #   expert_counts[l].sum() = 1000 × top_k = 8000
            #   expert_counts[l][e] / layer_tokens[l] = the frequency 
            #       that any token in layer l activates expert e 
            #       (this is what get_activation_profile() computes).
            # That ratio is what the rest of the system uses to decide 
            # which experts are "hot" (kept) vs. "cold" (skipped) — 
            # see thresholds τ_skip and τ_high.

            if not self.tracking:
                return
            # OLMoE / Mixtral both return (hidden_states, router_logits) as a tuple.
            if not (isinstance(output, tuple) and len(output) >= 2):
                return
            router_logits = output[1]   # get router_logits
            if router_logits is None:
                return
            with torch.no_grad():
                # top experts per token: shape (num_tokens, top_k)
                # router_logits: (20, 64)  →  top_indices: (20, 8)
                # example: [3, 17, 42, 5, 60, 11, 28, 7] experts were activated
                _, top_indices = torch.topk(router_logits, self.top_k, dim=-1)
                num_tokens = top_indices.shape[0]
                self.layer_tokens[layer_idx] += num_tokens
                
                # flatten and count how many times each expert was chosen overall
                flat = top_indices.reshape(-1)
                counts = torch.bincount(flat, minlength=self.num_experts)
                self.expert_counts[layer_idx] += counts.cpu().numpy()
        return hook_fn

    # ── Outputs ──
    def get_activation_profile(self) -> np.ndarray:
        """Returns ``(num_layers, num_experts)`` where each layer row sums to ``top_k``.

        Dividing by layer tokens (not total tokens * top_k) gives the probability
        that any given token in that layer activates expert e — with ``top_k``
        activations per token, each row sums to ``top_k``. Thresholds in the
        architecture doc (τ_skip=0.005, τ_high=0.05) are in these units.
        """
        if self.layer_tokens.sum() == 0:
            return np.zeros((self.num_layers, self.num_experts))
        denom = np.maximum(self.layer_tokens, 1).reshape(-1, 1)
        return self.expert_counts.astype(np.float64) / denom

    def get_activation_metadata(self) -> Dict[str, Any]:
        profile = self.get_activation_profile()
        return {
            "activation_freq": profile.tolist(),
            "total_tokens": int(self.total_tokens),
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }

    def summary(self) -> str:
        p = self.get_activation_profile()
        lines = [
            "ActivationTracker Summary:",
            f"  Tokens (max across layers): {self.total_tokens:,}",
            f"  Shape: {self.num_layers}L × {self.num_experts}E (top-{self.top_k})",
            f"  Per-layer sum (should equal top_k={self.top_k}): {p.sum(axis=1).mean():.3f}",
            f"  Mean freq: {p.mean():.4f} (uniform would be {self.top_k / self.num_experts:.4f})",
            f"  Max freq:  {p.max():.4f}    Min freq: {p.min():.6f}",
        ]
        return "\n".join(lines)
