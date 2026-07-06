#!/usr/bin/env python3
"""Original schematic of the semantic-entropy pipeline, built from this thesis's
own worked example (capital of Australia). Conveys the same idea as the method
figure in Farquhar et al. (2024) without reproducing its layout: counting
distinct wordings overstates uncertainty; grouping by meaning corrects it.

Layout: symmetric tree. Question + sampling on top, the two alternative
scorings side by side in the middle, the decision box at the bottom.
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

BLUE, ORANGE, GREEN, GREY = "#2563eb", "#d97706", "#16a34a", "#4b5563"
FILL_BLUE, FILL_ORANGE, FILL_GREEN, FILL_GREY = "#eef2ff", "#fff7ed", "#f0fdf4", "#f9fafb"

fig, ax = plt.subplots(figsize=(12.5, 6.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, body, fc, ec, title_fs=12.5, body_fs=11.5):
    """Rounded box with a bold colored title line and regular body text."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.7,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.6))
    if title:
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
                fontsize=title_fs, weight="bold", color=ec)
        ax.text(x + w / 2, y + (h - 9) / 2, body, ha="center", va="center",
                fontsize=body_fs, color="black", linespacing=1.5)
    else:
        ax.text(x + w / 2, y + h / 2, body, ha="center", va="center",
                fontsize=body_fs, color="black", linespacing=1.5)


def arrow(x1, y1, x2, y2, color=GREY):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.8, color=color,
                                 shrinkA=2, shrinkB=2))


# ---- Top row: question -> sample -------------------------------------------
box(4, 66, 21, 24, None,
    "Question\n\n“What is the capital\nof Australia?”",
    FILL_BLUE, BLUE)
arrow(26, 78, 33, 78, BLUE)
box(34, 60, 32, 33, "1. Sample the model $M=5$ times",
    "Canberra   ·   Canberra is the capital …\nSydney   ·   Melbourne   ·   Canberra",
    "white", BLUE)

# ---- Middle row: the two alternative scorings ------------------------------
arrow(45, 58, 27, 51, ORANGE)   # sample -> surface
arrow(55, 58, 73, 51, GREEN)    # sample -> semantic
box(3, 22, 46, 28, "2a. Count distinct wordings",
    "4 wording groups:\n"
    "{Canberra ×2}   {Canberra is the capital …}\n"
    "{Sydney}   {Melbourne}\n"
    "surface entropy $\\approx$ 1.33 nats: looks very unsure,\n"
    "because paraphrases count as different answers",
    FILL_ORANGE, ORANGE)
box(51, 22, 46, 28, "2b. Group by meaning (NLI entailment)",
    "3 meaning groups:\n"
    "{Canberra;  Canberra is the capital …;  Canberra}\n"
    "{Sydney}   {Melbourne}\n"
    "semantic entropy $\\approx$ 0.95 nats: the model has\n"
    "mostly settled on one answer",
    FILL_GREEN, GREEN)

# ---- Bottom row: decision ---------------------------------------------------
arrow(26, 20, 40, 13.5, ORANGE)
arrow(74, 20, 60, 13.5, GREEN)
box(27, 1, 46, 12, None,
    "3. Score:  high entropy $\\Rightarrow$ flag the answer as risky\n"
    "low entropy $\\Rightarrow$ treat it as reliable",
    FILL_GREY, GREY, body_fs=12)

fig.suptitle("Semantic entropy: measure uncertainty over meanings, not over wordings",
             fontsize=14.5, y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.95))
for d in (FIGS, TFIGS):
    d.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(d / f"method_schematic.{ext}", dpi=200, bbox_inches="tight")
print("wrote method_schematic.png to", FIGS, "and", TFIGS)
