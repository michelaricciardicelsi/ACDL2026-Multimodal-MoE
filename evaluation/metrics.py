"""
evaluation/metrics.py
MoE routing analysis: matplotlib + seaborn plots.

Generates and saves labeled charts for:
  1. Expert load distribution        — bar chart per layer
  2. Gate entropy distribution       — histogram across all MoE layers
  3. Routing frequency heatmap       — layers × experts heatmap
  4. Token allocation per expert     — bar chart (first layer as representative)

Usage:
    # Standalone demo with synthetic data:
    python evaluation/metrics.py

    # Import and use programmatically:
    from evaluation.metrics import plot_moe_statistics
    plot_moe_statistics(stats_dict, output_dir="output/plots")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")   # headless rendering — safe on Colab and servers
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Global styling
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":     150,
    "font.size":      12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "axes.titlepad":  10,
})

_PALETTE = sns.color_palette("muted")


# ---------------------------------------------------------------------------
# Individual plot helpers
# ---------------------------------------------------------------------------

def plot_expert_load(
    expert_utilization: np.ndarray,
    layer_name: str = "Layer",
    ax=None,
    save_path: Optional[str] = None,
) -> "plt.Axes":
    """Bar chart of expert load distribution (fraction of tokens routed to each expert)."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))

    num_experts = len(expert_utilization)
    expert_ids  = np.arange(num_experts)
    colors      = sns.color_palette("muted", num_experts)

    ax.bar(expert_ids, expert_utilization, color=colors, edgecolor="white", linewidth=0.5)
    ideal = 1.0 / num_experts
    ax.axhline(
        ideal, color="crimson", linestyle="--", linewidth=1.4,
        label=f"Ideal uniform ({ideal:.3f})",
    )
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Token Fraction")
    ax.set_title(f"{layer_name} — Expert Load Distribution")
    ax.set_xticks(expert_ids)
    ax.legend(fontsize=10)

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close()
    return ax


def plot_gate_entropy_histogram(
    entropy_values: List[float],
    ax=None,
    save_path: Optional[str] = None,
) -> "plt.Axes":
    """Histogram of gate entropy values across MoE layers."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.hist(
        entropy_values,
        bins=max(5, len(entropy_values) // 2),
        color=_PALETTE[1],
        edgecolor="white",
        linewidth=0.5,
    )
    if entropy_values:
        mean_e = float(np.mean(entropy_values))
        ax.axvline(mean_e, color="crimson", linestyle="--", linewidth=1.4,
                   label=f"Mean: {mean_e:.3f}")
        ax.legend(fontsize=10)

    ax.set_xlabel("Gate Entropy (nats)")
    ax.set_ylabel("Number of Layers")
    ax.set_title("Gate Entropy Distribution Across MoE Layers")

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close()
    return ax


def plot_routing_frequency_heatmap(
    routing_matrix: np.ndarray,
    layer_names: Optional[List[str]] = None,
    ax=None,
    save_path: Optional[str] = None,
) -> "plt.Axes":
    """Heatmap of routing frequencies: rows=layers, columns=experts."""
    num_layers, num_experts = routing_matrix.shape
    standalone = ax is None
    if standalone:
        fig_h = max(3.0, num_layers * 0.7)
        fig, ax = plt.subplots(figsize=(max(8, num_experts * 0.9), fig_h))

    yticklabels = layer_names or [f"Layer {i}" for i in range(num_layers)]
    xticklabels = [f"E{i}" for i in range(num_experts)]

    sns.heatmap(
        routing_matrix,
        ax=ax,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "Routing Frequency", "shrink": 0.8},
    )
    ax.set_xlabel("Expert")
    ax.set_ylabel("MoE Layer")
    ax.set_title("Expert Routing Frequency Heatmap")

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close()
    return ax


def plot_token_allocation(
    expert_utilization: np.ndarray,
    total_tokens: float = 1.0,
    layer_name: str = "Layer",
    ax=None,
    save_path: Optional[str] = None,
) -> "plt.Axes":
    """Bar chart of absolute token count per expert."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))

    token_counts = expert_utilization * total_tokens
    expert_ids   = np.arange(len(token_counts))
    colors       = sns.color_palette("muted", len(token_counts))

    bars = ax.bar(expert_ids, token_counts, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Token Count")
    ax.set_title(f"{layer_name} — Token Allocation per Expert")
    ax.set_xticks(expert_ids)

    # Annotate bars with count
    for bar, count in zip(bars, token_counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.01,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    if standalone:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close()
    return ax


# ---------------------------------------------------------------------------
# Combined dashboard
# ---------------------------------------------------------------------------

def plot_moe_statistics(
    stats: Dict[str, dict],
    output_dir: str = "output",
    show: bool = False,
):
    """Generate all MoE analysis plots and save PNGs to output_dir.

    Args:
        stats:      Dict mapping layer_name -> stats_dict.
                    Each stats_dict must have the keys returned by
                    SparseMoELayer.get_routing_stats().
        output_dir: Directory to save PNG files.
        show:       Call plt.show() after each plot (False = headless / Colab batch).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    layer_names    = list(stats.keys())
    entropy_values = [v["gate_entropy"]     for v in stats.values()]
    routing_matrix = np.array([v["routing_frequency"] for v in stats.values()])

    # 1. Per-layer expert load bar charts
    for layer_name, layer_stats in stats.items():
        util = np.array(layer_stats["expert_utilization"])
        plot_expert_load(
            util,
            layer_name=layer_name,
            save_path=str(out / f"expert_load_{layer_name}.png"),
        )
        if show:
            plt.show()

    # 2. Gate entropy histogram
    plot_gate_entropy_histogram(
        entropy_values,
        save_path=str(out / "gate_entropy_histogram.png"),
    )
    if show:
        plt.show()

    # 3. Routing frequency heatmap (all layers × all experts)
    if routing_matrix.ndim == 2 and routing_matrix.shape[0] > 0:
        plot_routing_frequency_heatmap(
            routing_matrix,
            layer_names=layer_names,
            save_path=str(out / "routing_frequency_heatmap.png"),
        )
        if show:
            plt.show()

    # 4. Token allocation for the first available layer
    if stats:
        first_key   = layer_names[0]
        first_stats = stats[first_key]
        util        = np.array(first_stats["expert_utilization"])
        total       = first_stats.get("total_tokens_routed", 1.0)
        plot_token_allocation(
            util,
            total_tokens=total,
            layer_name=first_key,
            save_path=str(out / "token_allocation_layer0.png"),
        )
        if show:
            plt.show()

    print(f"[metrics] All plots saved to: {out}/")


# ---------------------------------------------------------------------------
# Standalone demo with synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng         = np.random.default_rng(42)
    num_layers  = 6
    num_experts = 8

    # Simulate non-uniform (power-law-ish) utilisation
    fake_stats: Dict[str, dict] = {}
    for i in range(num_layers):
        raw  = rng.exponential(scale=1.0, size=num_experts)
        util = raw / raw.sum()
        probs = util / (util.sum() + 1e-9)
        entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        fake_stats[f"layer_{i * 2}"] = {
            "expert_utilization":   util,
            "routing_frequency":    util,
            "gate_entropy":         entropy,
            "pct_inactive_experts": float((util < 1e-4).mean() * 100),
            "total_tokens_routed":  float(rng.integers(2000, 8000)),
        }

    plot_moe_statistics(fake_stats, output_dir="output/demo_plots", show=False)
    print("Demo complete. Check output/demo_plots/")
