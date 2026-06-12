# mhqa/eda.py
"""
EDA figure generation. All plots use real data from Train.csv.
Functions are called from both the notebook and scripts/run_eda.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from mhqa.constants import (
    SUBSET_ORDER, SUBSET_LABELS, PALETTE, SHORT_LABELS,
    QUESTION_COL, ANSWER_COL, SUBSET_COL, LANG_FAMILY_MAP
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
BG = "#F8F9FA"


def _word_lengths(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["q_words"] = df[QUESTION_COL].str.split().str.len()
    df["a_words"] = df[ANSWER_COL].str.split().str.len()
    return df


def plot_eda_overview(df: pd.DataFrame, out_dir: Path) -> Path:
    """Four-panel EDA overview: distribution, Q-length, A-length boxplot, percentile heatmap."""
    df = _word_lengths(df)
    colors = [PALETTE[s] for s in SUBSET_ORDER]
    counts = df[SUBSET_COL].value_counts().reindex(SUBSET_ORDER)

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(BG)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A — sample counts
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(8), counts.values, color=colors, width=0.65, edgecolor="white")
    ax1.set_xticks(range(8)); ax1.set_xticklabels(SHORT_LABELS, fontsize=9)
    ax1.set_ylabel("Records", fontsize=10, fontweight="bold")
    ax1.set_title("A  Sample Distribution by Subset", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="x", length=0); ax1.set_facecolor(BG)
    ax1.set_ylim(0, counts.max() * 1.18)
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                 f"{val:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Panel B — question length by language family
    ax2 = fig.add_subplot(gs[0, 1])
    fam_colors = {"English": "#2D6A9F", "Akan": "#E05C2A",
                  "Luganda": "#4CAF82", "Swahili": "#8E5EA2", "Amharic": "#E8A838"}
    df["lang_family"] = df[SUBSET_COL].map(LANG_FAMILY_MAP)
    for fam, grp in sorted(df.groupby("lang_family")):
        ax2.hist(grp["q_words"].clip(upper=60), bins=25, alpha=0.55,
                 label=fam, color=fam_colors[fam], density=True, edgecolor="none")
    med = int(df["q_words"].median())
    ax2.axvline(med, color="#333", linestyle="--", linewidth=1.2)
    ax2.text(med + 0.5, ax2.get_ylim()[1] * 0.88, f"Median = {med}w", fontsize=8, color="#333")
    ax2.set_xlabel("Question Length (words)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax2.set_title("B  Question Length by Language Family", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax2.legend(fontsize=8.5, framealpha=0.8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.set_facecolor(BG)

    # Panel C — answer length boxplots
    ax3 = fig.add_subplot(gs[1, 0])
    data_bp = [df[df[SUBSET_COL] == s]["a_words"].clip(upper=350).values for s in SUBSET_ORDER]
    bp = ax3.boxplot(data_bp, patch_artist=True, widths=0.55,
                     medianprops=dict(color="white", linewidth=2.5),
                     flierprops=dict(marker="o", markersize=2.5, alpha=0.3, linestyle="none"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax3.set_xticks(range(1, 9)); ax3.set_xticklabels(SHORT_LABELS, fontsize=8.5)
    ax3.set_ylabel("Answer Length (words)", fontsize=10, fontweight="bold")
    ax3.set_title("C  Answer Length Distribution per Subset", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)
    ax3.tick_params(axis="x", length=0); ax3.set_facecolor(BG)

    # Panel D — percentile heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    hmap = []
    for s in SUBSET_ORDER:
        col = df[df[SUBSET_COL] == s]["a_words"]
        hmap.append([col.quantile(0.25), col.quantile(0.50),
                     col.quantile(0.75), col.quantile(0.95)])
    hmap_df = pd.DataFrame(hmap, index=SHORT_LABELS, columns=["p25", "p50", "p75", "p95"]).T
    sns.heatmap(hmap_df, ax=ax4, cmap="YlOrRd", annot=True, fmt=".0f",
                annot_kws={"size": 8}, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Word Count", "shrink": 0.8})
    ax4.set_title("D  Answer Length Percentiles (words)", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax4.tick_params(axis="x", rotation=30, labelsize=8)
    ax4.tick_params(axis="y", rotation=0)

    fig.suptitle("Exploratory Data Analysis — Training Set (N = 29,815)",
                 fontsize=14, fontweight="bold", y=0.99, color="#1A1A2E")

    out_path = out_dir / "01_subset_distribution.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved: {out_path}")
    return out_path


def plot_token_budget(df: pd.DataFrame, out_dir: Path) -> Path:
    """Bar charts showing what % of answers fit within different token budgets."""
    df = _word_lengths(df)
    budgets_words = [90, 180, 275]          # ≈128, 256, 384 mT5 tokens
    budget_labels = ["≤128 tok\n(90w)", "≤256 tok\n(180w)", "≤384 tok\n(275w)"]
    budget_colors = ["#E05C2A", "#2D6A9F", "#4CAF82"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)

    # Per-subset grouped bars
    ax = axes[0]
    x = np.arange(8); w = 0.25
    for i, (bw, bl, bc) in enumerate(zip(budgets_words, budget_labels, budget_colors)):
        pct = [(df[df[SUBSET_COL] == s]["a_words"] <= bw).mean() * 100 for s in SUBSET_ORDER]
        ax.bar(x + i * w, pct, w, label=bl, color=bc, alpha=0.85, edgecolor="white")
    ax.axhline(95, color="#333", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(7.6, 95.8, "95%", fontsize=8, color="#333")
    ax.set_xticks(x + w); ax.set_xticklabels(SHORT_LABELS, fontsize=9)
    ax.set_ylabel("% Answers Fitting Budget", fontsize=10, fontweight="bold")
    ax.set_title("A  Answer Coverage per Token Budget — by Subset",
                 fontsize=11, fontweight="bold", loc="left", pad=8)
    ax.legend(fontsize=9, framealpha=0.8, loc="lower right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_facecolor(BG); ax.set_ylim(0, 110); ax.tick_params(axis="x", length=0)

    # Overall coverage
    ax2 = axes[1]
    overall_pct = [(df["a_words"] <= bw).mean() * 100 for bw in budgets_words]
    bars2 = ax2.bar(range(3), overall_pct, color=budget_colors, alpha=0.85, edgecolor="white", width=0.5)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(budget_labels, fontsize=10)
    ax2.set_ylabel("% of All Answers", fontsize=10, fontweight="bold")
    ax2.set_title("B  Overall Coverage — Decision Point", fontsize=11, fontweight="bold", loc="left", pad=8)
    ax2.axhline(95, color="#333", linestyle="--", linewidth=1.2)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.set_facecolor(BG); ax2.set_ylim(60, 107)
    for bar, val in zip(bars2, overall_pct):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.annotate(
        "Selected: 384 tokens\n→ 99%+ overall\n→ 95%+ Akan (longest answers)",
        xy=(2, overall_pct[2]), xytext=(1.35, 85),
        arrowprops=dict(arrowstyle="->", color="#2D6A9F"),
        fontsize=8.5, color="#1A1A2E",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#E8F4FD", edgecolor="#2D6A9F"),
    )

    fig.suptitle("Token Budget Analysis — Justification for MAX_TARGET_LEN = 384",
                 fontsize=13, fontweight="bold", y=1.01, color="#1A1A2E")
    plt.tight_layout()

    out_path = out_dir / "02_length_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved: {out_path}")
    return out_path


def plot_per_subset_rouge(
    results_df: pd.DataFrame,
    overall: dict,
    experiment_name: str,
    out_dir: Path,
) -> Path:
    """Grouped bar chart of ROUGE-1, ROUGE-2, ROUGE-L per subset."""
    # results_df index = SUBSET_LABELS values, cols = rouge1, rouge2, rougeL
    subsets_present = [s for s in SUBSET_ORDER if SUBSET_LABELS[s] in results_df.index]
    labels = [SUBSET_LABELS[s].replace(" ", "\n") for s in subsets_present]
    r1 = results_df.loc[[SUBSET_LABELS[s] for s in subsets_present], "rouge1"].values
    r2 = results_df.loc[[SUBSET_LABELS[s] for s in subsets_present], "rouge2"].values
    rL = results_df.loc[[SUBSET_LABELS[s] for s in subsets_present], "rougeL"].values

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor(BG)
    x = np.arange(len(subsets_present)); w = 0.27
    ax.bar(x - w, r1, w, label="ROUGE-1", color="#2D6A9F", alpha=0.85, edgecolor="white")
    ax.bar(x,     r2, w, label="ROUGE-2", color="#4CAF82", alpha=0.85, edgecolor="white")
    ax.bar(x + w, rL, w, label="ROUGE-L", color="#E05C2A", alpha=0.85, edgecolor="white")
    ax.axhline(overall["rouge1"], color="#2D6A9F", linestyle="--", linewidth=1.2, alpha=0.6,
               label=f"Overall R1 = {overall['rouge1']:.3f}")
    ax.axhline(overall["rougeL"], color="#E05C2A", linestyle=":",  linewidth=1.2, alpha=0.6,
               label=f"Overall RL = {overall['rougeL']:.3f}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=10, fontweight="bold")
    ax.set_title(f"Per-Subset ROUGE Scores — {experiment_name}",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=8.5, framealpha=0.85, ncol=2)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_facecolor(BG); ax.tick_params(axis="x", length=0)
    plt.tight_layout()

    out_path = out_dir / f"per_subset_rouge_{experiment_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved: {out_path}")
    return out_path


def plot_training_curves(log_history: list, experiment_name: str, out_dir: Path) -> Path:
    """Three-panel training curve: loss, ROUGE-1, ROUGE-L."""
    train_logs = [(l["epoch"], l["loss"]) for l in log_history
                  if "loss" in l and "eval_loss" not in l]
    eval_logs  = [(l["epoch"], l["eval_loss"],
                   l.get("eval_rouge1", 0), l.get("eval_rougeL", 0))
                  for l in log_history if "eval_loss" in l]

    if len(train_logs) < 2 or len(eval_logs) < 1:
        print("Not enough log history to plot training curves.")
        return None

    t_ep, t_loss           = zip(*train_logs)
    e_ep, e_loss, e_r1, eL = zip(*eval_logs)

    fig = plt.figure(figsize=(15, 4))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t_ep, t_loss,  "o-",  color="#2D6A9F", linewidth=2, markersize=5, label="Train")
    ax1.plot(e_ep, e_loss, "s--", color="#E05C2A", linewidth=2, markersize=5, label="Validation")
    ax1.set_title("Loss", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(); ax1.set_facecolor(BG)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(e_ep, e_r1, "o-", color="#4CAF82", linewidth=2, markersize=5)
    ax2.set_title("Validation ROUGE-1", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1")
    ax2.set_facecolor(BG)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(e_ep, eL, "o-", color="#8E5EA2", linewidth=2, markersize=5)
    ax3.set_title("Validation ROUGE-L", fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("F1")
    ax3.set_facecolor(BG)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    fig.suptitle(f"Training Curves — {experiment_name}", fontweight="bold", y=1.01)
    out_path = out_dir / f"curves_{experiment_name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved: {out_path}")
    return out_path


def plot_leaderboard_progression(exp_df: pd.DataFrame, out_dir: Path) -> Path:
    """Line plot of leaderboard score across completed experiments."""
    completed = exp_df[exp_df["lb"].notna()].copy()
    if len(completed) < 2:
        print("Need ≥2 experiments with lb scores to plot progression.")
        return None

    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor(BG)
    x = range(len(completed))
    ax.plot(x, completed["lb"], "o-", color="#E8A838", linewidth=2.5, markersize=8, zorder=3)
    ax.fill_between(x, completed["lb"], completed["lb"].min() - 0.005,
                    alpha=0.12, color="#E8A838")
    best_pos = completed["lb"].values.argmax()
    ax.scatter([best_pos], [completed["lb"].iloc[best_pos]], s=160, zorder=5,
               color="#E05C2A", marker="*",
               label=f"Best: {completed['lb'].max():.4f} ({completed['id'].iloc[best_pos]})")
    ax.set_xticks(list(x)); ax.set_xticklabels(completed["id"].tolist(), fontsize=9)
    ax.set_ylabel("Leaderboard Score", fontsize=10, fontweight="bold")
    ax.set_title("Leaderboard Score Progression", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()

    out_path = out_dir / "leaderboard_progression.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
    print(f"Saved: {out_path}")
    return out_path
