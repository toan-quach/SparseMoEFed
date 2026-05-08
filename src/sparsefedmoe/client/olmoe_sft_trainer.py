"""OLMoE SFT trainer — NVFlare client entry point.

Runs under NVFlare's ScriptExecutor. Loads OLMoE (or any HF MoE model whose
config exposes ``num_experts`` / ``num_experts_per_tok`` and whose MoE blocks
return ``(hidden, router_logits)``), applies LoRA to configurable modules
(default: shared attention) with optional full-rank training of expert FFNs
and router gate, does ``LOCAL_EPOCHS`` of SFT, and emits an FLModel whose ``meta``
includes the per-expert activation profile for the server to consume.

All config lives in env vars so the same script works under both dev (tiny
model, toy data) and cluster (full OLMoE). When ``SPARSEFEDMOE_DUMMY_MODEL=1``
the trainer builds a small synthetic MoE model — useful for CI and smoke runs
that can't download a real checkpoint.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

import nvflare.client as flare
from nvflare.app_common.abstract.fl_model import FLModel, ParamsType

from sparsefedmoe.client.activation_tracker import ActivationTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Env-driven config ─────────────────────────────────────────────────────────
MODEL_NAME = os.path.expanduser(os.environ.get("SPARSEFEDMOE_MODEL", "allenai/OLMoE-1B-7B-0924"))
DATA_DIR = os.environ.get("SPARSEFEDMOE_DATA", "./data")
LOCAL_EPOCHS = int(os.environ.get("SPARSEFEDMOE_LOCAL_EPOCHS", "1"))
BATCH_SIZE = int(os.environ.get("SPARSEFEDMOE_BATCH_SIZE", "2"))
LEARNING_RATE = float(os.environ.get("SPARSEFEDMOE_LR", "2e-4"))
MAX_SEQ_LEN = int(os.environ.get("SPARSEFEDMOE_MAX_SEQ_LEN", "256"))
LORA_RANK = int(os.environ.get("SPARSEFEDMOE_LORA_RANK", "16"))
# NOTE: LORA_TARGETS controls which modules get LoRA adapters.
# Default: shared attention only (q_proj, k_proj, v_proj, o_proj).
# Expert FFN modules are: gate_proj, up_proj, down_proj. To also apply
# LoRA to expert FFNs, append them:
#   SPARSEFEDMOE_LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
# TRAIN_EXPERTS / TRAIN_ROUTER control full-rank training of expert FFNs
# and the MoE router gate respectively. When both LoRA and full-rank are
# enabled for the same module, it becomes dual-mode (LoRA adapters + base
# weight gradients) — valid but more memory-intensive.
LORA_TARGETS = os.environ.get("SPARSEFEDMOE_LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj")
TRAIN_EXPERTS_FULL_RANK = os.environ.get("SPARSEFEDMOE_TRAIN_EXPERTS", "1") == "1"
TRAIN_ROUTER_FULL_RANK = os.environ.get("SPARSEFEDMOE_TRAIN_ROUTER", "1") == "1"
DUMMY_MODEL = os.environ.get("SPARSEFEDMOE_DUMMY_MODEL", "0") == "1"
# Fraction of each client's dataset held out for per-round eval (perplexity /
# eval_loss). Set to 0.0 to disable eval (e.g. to reproduce the old behaviour).
EVAL_FRAC = float(os.environ.get("SPARSEFEDMOE_EVAL_FRAC", "0.1"))
# Coefficient on the router z-loss — penalises large router-logit magnitudes
# and stabilises training (ST-MoE §3). OLMoE's native aux loss (load-balance)
# is already applied by the model's forward; this is an *additional* term we
# track and optionally add to the objective. 0.0 == log-only.
Z_LOSS_COEF = float(os.environ.get("SPARSEFEDMOE_Z_LOSS_COEF", "0.0"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _build_dummy_moe_model() -> torch.nn.Module:
    """A minimal MoE-like model for smoke/CI runs with no HF download.

    Produces a model whose ``.config`` carries ``num_hidden_layers``,
    ``num_experts``, and ``num_experts_per_tok``, and whose submodules include
    a class named ``SparseMoeBlock`` that returns ``(hidden, router_logits)``
    from forward — exactly what ActivationTracker looks for.
    """
    from types import SimpleNamespace

    class SparseMoeBlock(torch.nn.Module):
        def __init__(self, dim: int, num_experts: int):
            super().__init__()
            self.router = torch.nn.Linear(dim, num_experts, bias=False)
            self.experts = torch.nn.ModuleList([
                torch.nn.ModuleDict({
                    "gate_proj": torch.nn.Linear(dim, dim, bias=False),
                    "up_proj": torch.nn.Linear(dim, dim, bias=False),
                    "down_proj": torch.nn.Linear(dim, dim, bias=False),
                })
                for _ in range(num_experts)
            ])

        def forward(self, x):  # x: (B*T, dim)
            logits = self.router(x)
            return x, logits

    class DummyMoE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_hidden_layers=2, num_experts=4, num_experts_per_tok=2,
                vocab_size=256, hidden_size=32,
            )
            self.embed = torch.nn.Embedding(256, 32)
            self.layers = torch.nn.ModuleList([SparseMoeBlock(32, 4) for _ in range(2)])
            self.lm_head = torch.nn.Linear(32, 256, bias=False)

        def forward(self, input_ids=None, labels=None, attention_mask=None, **_):
            h = self.embed(input_ids)  # (B, T, D)
            B, T, D = h.shape
            h = h.view(B * T, D)
            for blk in self.layers:
                h, _logits = blk(h)
            h = h.view(B, T, D)
            logits = self.lm_head(h)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]), labels.view(-1),
                    ignore_index=-100,
                )
            return SimpleNamespace(loss=loss, logits=logits)

        # HF-compatible shims
        def named_parameters(self, *a, **kw):  # noqa: D401
            return super().named_parameters(*a, **kw)

    return DummyMoE().to(DEVICE)


def setup_model():
    # if DUMMY_MODEL:
    #     logger.info("Using dummy MoE model (SPARSEFEDMOE_DUMMY_MODEL=1)")
    #     return _build_dummy_moe_model()

    logger.info("Loading model: %s", MODEL_NAME)
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    # output_router_logits must be True so the forward hook sees router_logits.
    if hasattr(cfg, "output_router_logits"):
        cfg.output_router_logits = True

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=cfg,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map=DEVICE,
    )

    # Optional LoRA — skipped on CPU-only dev runs to keep the smoke test fast.
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

        lora_target_modules = [m.strip() for m in LORA_TARGETS.split(",") if m.strip()]
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_RANK, lora_alpha=32, lora_dropout=0.0,
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, lora)
        logger.info("LoRA targets: %s", lora_target_modules)

        # Unfreeze expert FFN base weights for full-rank training.
        if TRAIN_EXPERTS_FULL_RANK:
            n_unfrozen = 0
            for name, param in model.named_parameters():
                if "experts" in name and "lora_" not in name:
                    param.requires_grad = True
                    n_unfrozen += 1
            logger.info("Unfroze %d expert FFN params for full-rank training", n_unfrozen)

        # Unfreeze router gate weights for full-rank training.
        if TRAIN_ROUTER_FULL_RANK:
            for name, param in model.named_parameters():
                if "mlp.gate" in name and "gate_proj" not in name and "experts" not in name:
                    param.requires_grad = True
                    logger.info("Unfroze router param: %s", name)

        # Summarise what is trainable and how.
        lora_params, full_rank_params, frozen_params = 0, 0, 0
        categories = {"shared_lora": 0, "expert_lora": 0, "expert_full": 0,
                       "router_full": 0, "other": 0}
        for name, param in model.named_parameters():
            n = param.numel()
            is_lora = "lora_" in name
            is_expert = "experts" in name
            is_router = ("mlp.gate" in name and "gate_proj" not in name
                         and "experts" not in name)
            if not param.requires_grad:
                frozen_params += n
                continue
            if is_lora:
                lora_params += n
                categories["expert_lora" if is_expert else "shared_lora"] += n
            else:
                full_rank_params += n
                if is_expert:
                    categories["expert_full"] += n
                elif is_router:
                    categories["router_full"] += n
                else:
                    categories["other"] += n

        total = lora_params + full_rank_params + frozen_params
        trainable = lora_params + full_rank_params
        logger.info("─── Training configuration ───")
        logger.info("  LoRA targets:        %s (rank=%d)", lora_target_modules, LORA_RANK)
        logger.info("  Shared attn (LoRA):  %s params", f"{categories['shared_lora']:,}")
        logger.info("  Expert FFN  (LoRA):  %s params", f"{categories['expert_lora']:,}")
        logger.info("  Expert FFN  (full):  %s params", f"{categories['expert_full']:,}")
        logger.info("  Router gate (full):  %s params", f"{categories['router_full']:,}")
        if categories["other"]:
            logger.info("  Other       (full):  %s params", f"{categories['other']:,}")
        logger.info("  Trainable: %s / %s (%.1f%%)",
                     f"{trainable:,}", f"{total:,}", 100 * trainable / total)
        logger.info("  Frozen:    %s", f"{frozen_params:,}")
    except Exception as e:  # noqa: BLE001
        logger.warning("LoRA setup skipped (%s); training full model.", e)

    return model


def setup_data(client_id: int):
    from datasets import Dataset, load_from_disk

    path = os.path.join(DATA_DIR, f"client_{client_id}")
    if os.path.exists(path):
        ds = load_from_disk(path)
        logger.info("Loaded %d samples from %s", len(ds), path)
        return ds
    logger.warning("No data at %s; using tiny dummy dataset.", path)
    texts = ["Hello world from client " + str(client_id)] * 8
    return Dataset.from_dict({"text": texts})


def tokenize_dataset(ds, tokenizer):
    def _tok(ex):
        out = tokenizer(
            ex["text"], truncation=True, max_length=MAX_SEQ_LEN,
            padding="max_length", return_tensors="np",
        )
        out["labels"] = out["input_ids"].copy()
        return out
    tokenized = ds.map(_tok, batched=True, remove_columns=ds.column_names)
    tokenized.set_format("torch")
    return tokenized


def extract_params(model) -> Dict[str, np.ndarray]:
    return {
        name: p.detach().cpu().numpy()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def load_params(model, params: Dict[str, np.ndarray]) -> None:
    sd = model.state_dict()
    for name, value in params.items():
        if name in sd:
            sd[name] = torch.from_numpy(np.asarray(value)).to(sd[name].device).to(sd[name].dtype)
    model.load_state_dict(sd, strict=False)


def _compute_router_z_loss(router_logits) -> Optional[torch.Tensor]:
    """ST-MoE router z-loss: mean of (logsumexp(router_logits))^2 across layers.

    ``router_logits`` is either ``None`` (dummy model) or a tuple of per-layer
    tensors of shape ``(B*T, num_experts)`` — the format HF MoE models return
    when ``output_router_logits=True``.
    """
    if router_logits is None:
        return None
    if isinstance(router_logits, torch.Tensor):
        layer_logits = [router_logits]
    else:
        layer_logits = [x for x in router_logits if isinstance(x, torch.Tensor)]
    if not layer_logits:
        return None
    terms = [torch.logsumexp(x, dim=-1).pow(2).mean() for x in layer_logits]
    return torch.stack(terms).mean()


def train_one_round(model, dataloader, tracker):
    model.train()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE,
    )
    tracker.start()
    total_loss, total_aux, total_z, total_steps = 0.0, 0.0, 0.0, 0
    aux_seen, z_seen = 0, 0
    for epoch in range(LOCAL_EPOCHS):
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            if loss is None:
                continue
            # Aux loss: HF MoE forward already folds this into ``loss`` when
            # output_router_logits=True, but exposes the raw tensor too.
            aux = getattr(out, "aux_loss", None)
            # z-loss: not provided by HF, compute from router_logits.
            router_logits = getattr(out, "router_logits", None)
            z = _compute_router_z_loss(router_logits)

            final = loss
            if z is not None and Z_LOSS_COEF > 0.0:
                final = final + Z_LOSS_COEF * z.to(loss.device)

            final.backward()
            optim.step()
            optim.zero_grad()
            total_loss += float(loss.item())
            total_steps += 1
            if aux is not None:
                total_aux += float(aux.item())
                aux_seen += 1
            if z is not None:
                total_z += float(z.item())
                z_seen += 1
        logger.info("  Epoch %d loss=%.4f", epoch + 1, total_loss / max(total_steps, 1))
    tracker.stop()
    logger.info(tracker.summary())
    return {
        "train_loss": total_loss / max(total_steps, 1),
        "aux_loss": (total_aux / aux_seen) if aux_seen else None,
        "z_loss": (total_z / z_seen) if z_seen else None,
        "steps": total_steps,
    }


def evaluate(model, dataloader) -> Optional[Dict[str, float]]:
    """No-grad pass over ``dataloader``. Returns eval_loss + perplexity.

    Token-weighted: each batch contributes its loss × number of non-ignored
    label tokens, so the final loss matches ``CE`` over the whole eval set
    regardless of batch-size variation.
    """
    if dataloader is None:
        return None
    model.eval()
    loss_sum, token_sum = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.get("labels")
            if labels is None:
                continue
            # Count tokens the same way HF's CE loss does: positions where
            # labels != -100.  We intentionally do NOT intersect with
            # attention_mask here — the model's loss already ignores -100
            # positions, and mixing in a second mask would make our
            # loss × n_tokens product inconsistent with the model's mean.
            n_tokens = int((labels != -100).sum().item())
            if n_tokens == 0:
                continue
            out = model(**batch)
            if out.loss is None:
                continue
            loss_sum += float(out.loss.item()) * n_tokens
            token_sum += n_tokens
    model.train()
    if token_sum == 0:
        return None
    avg = loss_sum / token_sum
    return {
        "eval_loss": avg,
        "perplexity": float(np.exp(min(avg, 50.0))),  # clamp to avoid overflow
        "eval_tokens": token_sum,
    }


def _unwrap_for_tracker(model) -> torch.nn.Module:
    """Reach the underlying HF/MoE module through PEFT wrappers if present."""
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    if hasattr(model, "base_model"):
        return model.base_model
    return model


def main():
    flare.init()
    client_name = flare.get_site_name()
    try:
        # NVFlare site names are 1-indexed (site-1, site-2, ...) but the data
        # partitions are 0-indexed (client_0, client_1, ...). Map accordingly.
        client_id = int(client_name.split("-")[-1]) - 1
    except (ValueError, IndexError):
        client_id = 0
    if client_id < 0:
        client_id = 0
    logger.info("Client '%s' (id=%d) starting", client_name, client_id)

    model = setup_model()
    tokenizer = (
        AutoTokenizer.from_pretrained(MODEL_NAME)
        if not DUMMY_MODEL
        else _DummyTokenizer()
    )
    if not DUMMY_MODEL and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = setup_data(client_id)
    tokenized = tokenize_dataset(ds, tokenizer) if not DUMMY_MODEL else _dummy_tokenize(ds)

    # Hold out the tail of the client's data for eval. We want the same split
    # every round so eval_loss is comparable across rounds — ``select`` on a
    # contiguous range gives a deterministic held-out set without needing a
    # seeded shuffle.
    n_total = len(tokenized)
    n_eval = int(n_total * EVAL_FRAC) if EVAL_FRAC > 0 else 0
    if n_eval > 0 and n_total - n_eval >= 1:
        train_ds = tokenized.select(range(n_total - n_eval))
        eval_ds = tokenized.select(range(n_total - n_eval, n_total))
    else:
        train_ds = tokenized
        eval_ds = None

    dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=default_data_collator,
    )
    eval_dl = (
        DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False,
                   collate_fn=default_data_collator)
        if eval_ds is not None else None
    )
    logger.info("Data split: train=%d eval=%d", len(train_ds), n_eval)

    tracker = ActivationTracker(_unwrap_for_tracker(model))
    print(f"[DIAG] tracker shape: ({tracker.num_layers}, {tracker.num_experts}), top_k={tracker.top_k}", flush=True)

    while flare.is_running():
        input_model = flare.receive()
        logger.info("Received global model (round %s)", getattr(input_model, "current_round", "?"))

        if input_model.params:
            load_params(model, input_model.params)

        # Surface server-issued floor tiers so the encoder filter can apply them.
        floor_tiers = (input_model.meta or {}).get("floor_tiers")
        if floor_tiers is not None:
            # Persist to env var so the encoder filter (a separate class) can read
            # via its own update path. Simplest cross-object channel available
            # under ScriptExecutor without tighter NVFlare wiring.
            os.environ["SPARSEFEDMOE_FLOOR_TIERS"] = _encode_floor_tiers(floor_tiers)

        tracker.reset()
        train_stats = train_one_round(model, dl, tracker)
        eval_stats = evaluate(model, eval_dl)

        updated = extract_params(model)
        if input_model.params:
            param_diffs = {
                name: updated[name] - np.asarray(input_model.params[name])
                if name in input_model.params
                else updated[name]
                for name in updated
            }
        else:
            param_diffs = updated

        meta: Dict[str, Any] = {
            "activation_profile": tracker.get_activation_metadata(),
            "loss": float(train_stats["train_loss"]),
            "num_samples": len(train_ds),
        }
        # Router losses: log-only unless Z_LOSS_COEF > 0. ``aux_loss`` is
        # always reported because OLMoE already applies it inside ``out.loss``
        # and downstream analyses will want it broken out.
        if train_stats["aux_loss"] is not None:
            meta["aux_loss"] = float(train_stats["aux_loss"])
        if train_stats["z_loss"] is not None:
            meta["z_loss"] = float(train_stats["z_loss"])
        if eval_stats is not None:
            meta["eval_loss"] = float(eval_stats["eval_loss"])
            meta["perplexity"] = float(eval_stats["perplexity"])
            meta["eval_tokens"] = int(eval_stats["eval_tokens"])

        output = FLModel(
            params_type=ParamsType.DIFF,
            params=param_diffs,
            meta=meta,
        )
        flare.send(output)
        logger.info(
            "Sent update: %d params, loss=%.4f aux=%s z=%s ppl=%s tokens=%d",
            len(param_diffs),
            train_stats["train_loss"],
            _fmt(train_stats["aux_loss"]),
            _fmt(train_stats["z_loss"]),
            _fmt(eval_stats["perplexity"] if eval_stats else None, fmt=".2f"),
            tracker.total_tokens,
        )


def _fmt(v, fmt: str = ".4f") -> str:
    """Compact formatter that handles ``None`` so log lines stay readable."""
    if v is None:
        return "n/a"
    return f"{v:{fmt}}"


# ── Dummy-mode helpers ──
class _DummyTokenizer:
    pad_token = None
    eos_token = 0

    def __call__(self, texts, **_):
        if isinstance(texts, str):
            texts = [texts]
        ids = np.zeros((len(texts), 8), dtype=np.int64)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t.encode()[:8]):
                ids[i, j] = ch
        return {"input_ids": ids, "attention_mask": np.ones_like(ids)}


def _dummy_tokenize(ds):
    from datasets import Dataset as _D
    tok = _DummyTokenizer()
    rows = []
    for row in ds:
        out = tok(row["text"])
        rows.append({
            "input_ids": out["input_ids"][0].tolist(),
            "attention_mask": out["attention_mask"][0].tolist(),
            "labels": out["input_ids"][0].tolist(),
        })
    ds2 = _D.from_list(rows)
    ds2.set_format("torch")
    return ds2


def _encode_floor_tiers(tiers) -> str:
    # tiers is either a list of [l,e,tier] or a dict; emit CSV.
    entries = tiers.items() if isinstance(tiers, dict) else [(e[0], e[1], e[2]) for e in tiers]
    return ";".join(f"{l},{e},{t}" for (l, e, t) in entries) if not isinstance(tiers, dict) else ";".join(
        f"{k[0]},{k[1]},{v}" for k, v in tiers.items()
    )


if __name__ == "__main__":
    main()
