"""Model-introspection helpers shared by filters, aggregator, and trainer.

These predicates classify a parameter name as belonging to an MoE expert FFN,
the MoE router/gate, or generic shared params (attention, norms, embeddings).
They are pattern-based (name-string), so they work equally well on raw HF
modules and on PEFT/LoRA-wrapped modules whose parameter names preserve the
upstream `model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight` pattern.
"""

from __future__ import annotations

from typing import Optional, Tuple


_EXPERT_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


def parse_expert_indices(param_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse (layer_idx, expert_idx) from an MoE expert parameter name.

    Matches names like `...layers.{L}...experts.{E}.{gate_proj|up_proj|down_proj}.weight`.
    Also tolerates PEFT LoRA-wrapped names that splice `base_model.model.` prefix and
    `lora_A.default.weight` / `lora_B.default.weight` suffix — the parse still works
    because we locate by the `layers` and `experts` tokens anywhere in the path.
    """
    parts = param_name.split(".")
    try:
        if "experts" in parts and "layers" in parts:
            layers_pos = parts.index("layers")
            experts_pos = parts.index("experts")
            return int(parts[layers_pos + 1]), int(parts[experts_pos + 1])
    except (ValueError, IndexError):
        pass
    return None, None


def is_expert_param(param_name: str) -> bool:
    """True for per-expert FFN weights (gate_proj / up_proj / down_proj)."""
    if "experts" not in param_name:
        return False
    return any(proj in param_name for proj in _EXPERT_PROJ_NAMES)


def is_router_param(param_name: str) -> bool:
    """True for MoE gating/router weights.

    OLMoE and Mixtral both use `...mlp.gate.weight` for the router. Under PEFT's
    ``modules_to_save=["gate"]`` the saved copy appears as
    `...mlp.gate.modules_to_save.default.weight`. We match both.
    """
    if "mlp.gate" not in param_name and ".gate." not in param_name:
        return False
    # Rule out expert-gate_proj (different module): gate_proj lives under `experts.`.
    if "experts" in param_name or "gate_proj" in param_name:
        return False
    return True


def classify_param(param_name: str) -> str:
    """Return one of {"expert", "router", "shared"} for any parameter name."""
    if is_expert_param(param_name):
        return "expert"
    if is_router_param(param_name):
        return "router"
    return "shared"
