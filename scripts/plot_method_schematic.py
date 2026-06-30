#!/usr/bin/env python3
"""Original schematic of the semantic-entropy pipeline, built from this thesis's
own worked example (capital of Australia). Conveys the same idea as the method
figure in Farquhar et al. (2024) without reproducing its layout: counting
distinct wordings overstates uncertainty; grouping by meaning corrects it.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "results/figures"
TFIGS = ROOT / "thesis/figures"

BLUE, ORANGE, GREEN, GREY = "#2563eb", "#d97706", "#16a34a", "#6b7280"
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=GREY, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            weight="bold" if bold else "normal", wrap=True)


def arrow(x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=1.5, color=color))


# 1. Question
box(2, 70, 22, 12, "Question\n\"What is the capital\nof Australia?\"", fc="#eef2ff", ec=BLUE, fs=9, bold=True)
# 2. Sample
arrow(24, 76, 30, 76, BLUE)
box(30, 64, 22, 24,
    "Sample the model\n$M$ times:\n\nCanberra\nCanberra is the capital…\nSydney\nMelbourne\nCanberra",
    fc="white", ec=BLUE, fs=8)
ax.text(41, 90, "1. Generate", ha="center", fontsize=10, weight="bold", color=BLUE)

# Split into two paths
arrow(52, 80, 62, 90, ORANGE)
arrow(52, 72, 62, 40, GREEN)

# 3a. Surface path (top)
ax.text(81, 96, "Count distinct wordings (surface)", ha="center", fontsize=10, weight="bold", color=ORANGE)
box(62, 74, 36, 18,
    "4 groups: {Canberra×2}, {Canberra is…}, {Sydney}, {Melbourne}\n"
    r"$\Rightarrow$ surface entropy $\approx 1.33$ nats",
    fc="#fff7ed", ec=ORANGE, fs=8.5)
ax.text(80, 70.5, "looks very unsure (paraphrase inflates it)", ha="center", fontsize=8, style="italic", color=ORANGE)

# 3b. Semantic path (bottom)
ax.text(81, 56, "2. Group by meaning (NLI entailment)", ha="center", fontsize=10, weight="bold", color=GREEN)
box(62, 30, 36, 22,
    "3 meaning-groups:\n{Canberra, Canberra is the\ncapital…, Canberra}, {Sydney}, {Melbourne}\n"
    r"$\Rightarrow$ semantic entropy $\approx 0.95$ nats",
    fc="#f0fdf4", ec=GREEN, fs=8.5)
ax.text(80, 26.5, "more honest: the model mostly settled on one meaning", ha="center", fontsize=8, style="italic", color=GREEN)

# 4. Decision
arrow(80, 30, 80, 18, GREEN)
arrow(80, 74, 80, 64, ORANGE)
box(34, 6, 46, 12,
    "3. Score: high entropy → flag answer as risky;\nlow entropy → treat as reliable",
    fc="#f9fafb", ec=GREY, fs=9, bold=True)
arrow(40, 64, 52, 18, GREY)

fig.suptitle("Semantic entropy: measure uncertainty over meanings, not over wordings",
             fontsize=12, y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.97))
for d in (FIGS, TFIGS):
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"method_schematic.{ext}", dpi=150, bbox_inches="tight")
print("wrote method_schematic.png to", FIGS, "and", TFIGS)
