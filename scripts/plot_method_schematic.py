#!/usr/bin/env python3
"""Draw the thesis's constructed five-answer example as a vector schematic.

Normalized wording counts are (2, 1, 1, 1); meaning counts are (3, 1, 1).
The diagram illustrates count-based entropy, not a correctness guarantee.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, LINE = "#233345", "#586575", "#CFD7DF"
BLUE, TEAL = "#315F86", "#23776F"

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.2, 3.55))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax.set(xlim=(0, 100), ylim=(0, 100))
ax.axis("off")


def box(x, y, width, height, fill, edge=LINE, radius=1.2):
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=.7, facecolor=fill, edgecolor=edge))


def text(x, y, label, size=8.3, color=INK, weight="normal", ha="left"):
    ax.text(x, y, label, fontsize=size, color=color, weight=weight,
            ha=ha, va="center", linespacing=1.2)


def arrow(x1, y1, x2, y2, color):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
        linewidth=.9, color=color, shrinkA=0, shrinkB=0))


def frequency(x, y, count, color):
    # Filled marks encode counts; outlines keep the five-sample denominator visible.
    for i in range(5):
        ax.plot(x + i * 1.6, y, marker="o", markersize=3.4,
                markerfacecolor=color if i < count else "white",
                markeredgecolor=color if i < count else LINE, markeredgewidth=.6)
    text(x + 8.3, y, f"{count}/5", size=7.6, color=MUTED)


text(1, 96, "1  Sample five answers", size=9.5, weight="bold")
text(99, 96, "What is the capital of Australia?", size=9.5, ha="right")
answers = [(1, 16, "Canberra"), (19, 28, "Canberra is the\ncapital of Australia."),
           (49, 15, "Sydney"), (66, 15, "Melbourne"), (83, 16, "Canberra")]
for x, width, label in answers:
    box(x, 79, width, 12, "#F3F6F8")
    text(x + width / 2, 85, label, size=8.0, ha="center")
# Both branches use the same five sampled answers, so every box feeds the rail.
for x, width, _ in answers:
    ax.plot([x + width / 2, x + width / 2], [79, 75.5], color=LINE, lw=.9)
first_centre = answers[0][0] + answers[0][1] / 2
last_centre = answers[-1][0] + answers[-1][1] / 2
ax.plot([min(first_centre, 25), max(last_centre, 75)], [75.5, 75.5], color=LINE, lw=.9)
arrow(25, 75.5, 25, 71.5, BLUE)
arrow(75, 75.5, 75, 71.5, TEAL)

for x, color, fill, title, subtitle in [
    (1, BLUE, "#F1F5FA", "2a  Count normalized wordings", "4 groups · paraphrases stay separate"),
    (52, TEAL, "#EFF7F5", "2b  Group by meaning", "3 groups · bidirectional entailment"),
]:
    box(x, 15, 47, 56, "white")
    box(x, 57, 47, 14, fill, edge=fill)
    text(x + 2.4, 66, title, size=9.0, color=color, weight="bold")
    text(x + 2.4, 60.6, subtitle, size=7.5, color=MUTED)

wordings = [(51, "Canberra", 2), (42, "Canberra is the\ncapital of Australia.", 1),
            (33, "Sydney", 1), (24, "Melbourne", 1)]
for y, label, count in wordings:
    text(3.4, y, label, size=8.0)
    frequency(35, y, count, BLUE)
meanings = [(48, "Canberra / Canberra is the\ncapital of Australia.", 3),
            (36, "Sydney", 1), (24, "Melbourne", 1)]
for y, label, count in meanings:
    text(54.4, y, label, size=8.0)
    frequency(86, y, count, TEAL)


def entropy(counts):
    return -sum((n / 5) * math.log(n / 5) for n in counts)


# Values are calculated from the displayed frequencies, not entered by hand.
for x, color, name, counts in [(1, BLUE, "Surface entropy", [2, 1, 1, 1]),
                                (52, TEAL, "Discrete semantic entropy", [3, 1, 1])]:
    ax.plot([x, x + 47], [13, 13], color=color, lw=1.2)
    text(x + 1, 9.2, name, size=8.5, color=color, weight="bold")
    text(x + 46, 9.2, f"{entropy(counts):.2f} nats", size=10.0,
         color=color, weight="bold", ha="right")
text(50, 2.3, "Lower entropy means more agreement, not necessarily a correct answer.",
     size=8.0, color=MUTED, ha="center")

directory = ROOT / "thesis/figures"
directory.mkdir(parents=True, exist_ok=True)
fig.savefig(directory / "method_schematic.pdf", dpi=240,
            facecolor="white", metadata={"Creator": "Matplotlib"})
plt.close(fig)
print("Wrote thesis/figures/method_schematic.pdf")
