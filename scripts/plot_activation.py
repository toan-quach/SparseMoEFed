#!/usr/bin/env python3
"""Visualize MoE expert activation frequencies from SparseFedMoE runs.

Reads either:
  1. metrics.json from an NVFlare run (contains per-round, per-client profiles)
  2. A single-client JSON from run_local_sft_with_activation.py

Figures produced:
  - heatmap:  Per-client activation heatmap (layers × experts) for a given round
  - grid:     Side-by-side heatmaps comparing all clients in a round
  - timeline: Per-expert frequency across rounds (one subplot per layer)

Usage:
  # NVFlare metrics — all figures for round 1
  python scripts/plot_activation.py metrics.json --round 1 --out ./figures

  # NVFlare metrics — timeline across all rounds
  python scripts/plot_activation.py metrics.json --type timeline --out ./figures

  # Single-client JSON from run_local_sft_with_activation.py
  python scripts/plot_activation.py /tmp/act_client0.json --out ./figures

  # Multiple single-client JSONs — grid comparison
  python scripts/plot_activation.py /tmp/act_client0.json /tmp/act_client1.json --out ./figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def _load_json(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _is_nvflare_metrics(data: Dict[str, Any]) -> bool:
    return "rounds" in data and isinstance(data["rounds"], list)


def _extract_round_profiles(
    rounds: List[Dict], round_idx: Optional[int],
) -> Tuple[int, Dict[str, np.ndarray]]:
    """Return (round_number, {client_name: profile_array}) for a given round."""
    candidates = [r for r in rounds if "activation_profiles" in r]
    if not candidates:
        print("No activation profiles found in metrics.json. "
              "Was the run done with a version that records them?", file=sys.stderr)
        sys.exit(1)
    if round_idx is not None:
        match = [r for r in candidates if r["round"] == round_idx]
        if not match:
            avail = [r["round"] for r in candidates]
            print(f"Round {round_idx} not found. Available: {avail}", file=sys.stderr)
            sys.exit(1)
        entry = match[0]
    else:
        entry = candidates[-1]
    profiles = {
        c: np.asarray(v, dtype=np.float64)
        for c, v in entry["activation_profiles"].items()
    }
    return entry["round"], profiles


def _extract_timeline(
    rounds: List[Dict],
) -> Tuple[List[int], Dict[str, List[np.ndarray]]]:
    """Return (round_numbers, {client: [profile_per_round]})."""
    round_nums: List[int] = []
    per_client: Dict[str, List[np.ndarray]] = {}
    for r in rounds:
        if "activation_profiles" not in r:
            continue
        round_nums.append(r["round"])
        for c, v in r["activation_profiles"].items():
            per_client.setdefault(c, []).append(np.asarray(v, dtype=np.float64))
    return round_nums, per_client


# ── Plot functions ──────────────────────────────────────────────────────────

def plot_heatmap(
    profile: np.ndarray,
    title: str = "Expert Activation Frequency",
    ax: Optional[plt.Axes] = None,
    show_values: bool = True,
) -> plt.Axes:
    """Single heatmap: layers (rows) × experts (columns)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, profile.shape[1] * 0.6), max(3, profile.shape[0] * 0.5)))

    num_layers, num_experts = profile.shape
    vmin = max(profile[profile > 0].min(), 1e-4) if (profile > 0).any() else 1e-4
    im = ax.imshow(profile, aspect="auto", cmap="YlOrRd", norm=LogNorm(vmin=vmin, vmax=profile.max()))
    plt.colorbar(im, ax=ax, label="Activation frequency (log scale)")

    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")
    ax.set_xticks(range(num_experts))
    ax.set_yticks(range(num_layers))
    ax.set_title(title)

    if show_values and num_experts <= 16 and num_layers <= 32:
        for i in range(num_layers):
            for j in range(num_experts):
                val = profile[i, j]
                color = "white" if val > profile.max() * 0.5 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color)
    return ax


def plot_grid(
    profiles: Dict[str, np.ndarray],
    round_num: int,
    out_dir: Path,
) -> Path:
    """Side-by-side heatmaps for all clients in one round."""
    n = len(profiles)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    sample = next(iter(profiles.values()))
    fig_w = max(6, sample.shape[1] * 0.6) * cols
    fig_h = max(3, sample.shape[0] * 0.5) * rows + 1
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, (client, prof) in enumerate(sorted(profiles.items())):
        r, c = divmod(idx, cols)
        plot_heatmap(prof, title=client, ax=axes[r][c], show_values=n <= 4)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Activation Frequency — Round {round_num}", fontsize=14, y=1.02)
    fig.tight_layout()
    path = out_dir / f"activation_grid_round{round_num}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_timeline(
    round_nums: List[int],
    per_client: Dict[str, List[np.ndarray]],
    out_dir: Path,
) -> Path:
    """Line plot: expert frequency over rounds, one subplot per layer."""
    sample = next(iter(per_client.values()))[0]
    num_layers, num_experts = sample.shape

    fig, axes = plt.subplots(
        num_layers, 1,
        figsize=(max(8, len(round_nums) * 0.8), 3 * num_layers),
        sharex=True, squeeze=False,
    )
    cmap = plt.cm.tab20
    colors = [cmap(i / max(num_experts - 1, 1)) for i in range(num_experts)]

    for layer_idx in range(num_layers):
        ax = axes[layer_idx][0]
        for client_name, profiles in sorted(per_client.items()):
            freqs_by_expert = np.array([p[layer_idx] for p in profiles])
            for e in range(num_experts):
                label = f"E{e}" if len(per_client) == 1 else f"{client_name} E{e}"
                linestyle = "-" if len(per_client) == 1 else None
                ax.plot(
                    round_nums[:len(profiles)], freqs_by_expert[:, e],
                    color=colors[e], linestyle=linestyle, alpha=0.7,
                    label=label if layer_idx == 0 else None,
                    linewidth=1.5,
                )
        ax.set_ylabel(f"Layer {layer_idx}")
        ax.grid(True, alpha=0.3)

    axes[-1][0].set_xlabel("Round")
    fig.suptitle("Expert Activation Frequency Over Rounds", fontsize=14)
    if num_experts <= 16:
        axes[0][0].legend(
            bbox_to_anchor=(1.02, 1), loc="upper left",
            fontsize=7, ncol=max(1, num_experts // 8),
        )
    fig.tight_layout()
    path = out_dir / "activation_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="metrics.json or single-client JSON file(s)")
    parser.add_argument("--round", type=int, default=None, help="Round to visualize (default: last)")
    parser.add_argument("--type", choices=["heatmap", "grid", "timeline", "all"], default="all")
    parser.add_argument("--out", type=str, default="./figures", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load_json(args.inputs[0])

    if _is_nvflare_metrics(data):
        rounds = data["rounds"]

        if args.type in ("heatmap", "grid", "all"):
            round_num, profiles = _extract_round_profiles(rounds, args.round)
            if args.type in ("grid", "all"):
                p = plot_grid(profiles, round_num, out_dir)
                print(f"Grid:     {p}")
            if args.type in ("heatmap", "all"):
                for client, prof in sorted(profiles.items()):
                    fig, ax = plt.subplots(figsize=(max(6, prof.shape[1] * 0.6), max(3, prof.shape[0] * 0.5)))
                    plot_heatmap(prof, title=f"{client} — Round {round_num}", ax=ax)
                    fig.tight_layout()
                    p = out_dir / f"activation_{client}_round{round_num}.png"
                    fig.savefig(p, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"Heatmap:  {p}")

        if args.type in ("timeline", "all"):
            round_nums, per_client = _extract_timeline(rounds)
            if round_nums:
                p = plot_timeline(round_nums, per_client, out_dir)
                print(f"Timeline: {p}")
            else:
                print("No multi-round data for timeline.", file=sys.stderr)

    else:
        all_profiles: Dict[str, np.ndarray] = {}
        for inp in args.inputs:
            d = _load_json(inp) if inp != args.inputs[0] else data
            prof = d.get("activation_profile", {})
            freq = prof.get("activation_freq")
            if freq is None:
                print(f"No activation_freq in {inp}", file=sys.stderr)
                continue
            label = f"client_{d.get('client_id', Path(inp).stem)}"
            all_profiles[label] = np.asarray(freq, dtype=np.float64)

        if not all_profiles:
            print("No profiles to plot.", file=sys.stderr)
            sys.exit(1)

        if len(all_profiles) == 1:
            name, prof = next(iter(all_profiles.items()))
            fig, ax = plt.subplots(figsize=(max(6, prof.shape[1] * 0.6), max(3, prof.shape[0] * 0.5)))
            plot_heatmap(prof, title=name, ax=ax)
            fig.tight_layout()
            p = out_dir / f"activation_{name}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Heatmap: {p}")
        else:
            n = len(all_profiles)
            cols = min(n, 3)
            rows = (n + cols - 1) // cols
            sample = next(iter(all_profiles.values()))
            fig_w = max(6, sample.shape[1] * 0.6) * cols
            fig_h = max(3, sample.shape[0] * 0.5) * rows + 1
            fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
            for idx, (name, prof) in enumerate(sorted(all_profiles.items())):
                r, c = divmod(idx, cols)
                plot_heatmap(prof, title=name, ax=axes[r][c], show_values=n <= 4)
            for idx in range(n, rows * cols):
                r, c = divmod(idx, cols)
                axes[r][c].set_visible(False)
            fig.suptitle("Activation Frequency Comparison", fontsize=14, y=1.02)
            fig.tight_layout()
            p = out_dir / "activation_comparison.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Grid: {p}")


if __name__ == "__main__":
    main()
