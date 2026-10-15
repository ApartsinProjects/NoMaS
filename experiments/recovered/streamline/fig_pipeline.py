# -*- coding: utf-8 -*-
"""ADReal construction pipeline flowchart (Figure 1). Uniform box widths, subtitles wrapped to fit
inside their boxes, operations on the right of each arrow, drop counts in red on the left."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAV = "#14385c"; INK = "#1c2733"; GREY = "#5b6673"; RED = "#c23b52"; BAND = "#eef1f4"
W = 5.9            # uniform box width
CX = 3.35          # box centre x (left half); operations go in the right half
OPX = 6.55         # left edge of operations text
fig, ax = plt.subplots(figsize=(9.6, 7.3))
ax.set_xlim(0, 10); ax.set_ylim(0, 10.2); ax.axis("off")


def box(cy, title, subs, fc="white", ec=NAV, tc=INK, sc=GREY):
    h = 0.62 + 0.34 * len(subs)
    ax.add_patch(FancyBboxPatch((CX - W / 2, cy - h / 2), W, h,
                 boxstyle="round,pad=0.02,rounding_size=0.10",
                 linewidth=1.6, edgecolor=ec, facecolor=fc))
    ax.text(CX, cy + h / 2 - 0.28, title, ha="center", va="center",
            fontsize=12, fontweight="bold", color=tc)
    for i, s in enumerate(subs):
        ax.text(CX, cy + h / 2 - 0.60 - 0.30 * i, s, ha="center", va="center",
                fontsize=8.3, color=sc)
    return h


def arrow(y0, y1):
    ax.add_patch(FancyArrowPatch((CX, y0), (CX, y1), arrowstyle="-|>",
                 mutation_scale=16, lw=1.7, color=INK))


def ops(cy, lines):
    for i, s in enumerate(lines):
        ax.text(OPX, cy + 0.30 * (len(lines) - 1) / 2 - 0.30 * i, s,
                ha="left", va="center", fontsize=8.2, color=INK)


def drop(y, txt):
    ax.text(CX - W / 2 - 0.25, y, txt, ha="right", va="center",
            fontsize=8.8, color=RED, fontweight="bold")


# --- tiers (centres) ---
y1, y2, y3, y4 = 9.35, 7.05, 4.55, 1.85
h1 = box(y1, "Source collections",
         ["ADBench 35  \u00b7  DAMI 16  \u00b7  OddBench 187",
          "OvrBench 754 (tabular)  \u00b7  TSB-AD (time series)"], fc=BAND, ec=GREY)
h2 = box(y2, "174 candidate tasks", ["tabular + multivariate time-series"])
h3 = box(y3, "173 constructed tasks",
         ["train normals  \u00b7  validation normals (label-free)",
          "test = held-out normals + hard anomalies"])
h4 = box(y4, "151 in-benchmark tasks",
         ["time series 38  \u00b7  OvrBench 66  \u00b7  OddBench 39",
          "ADBench 5  \u00b7  DAMI 3"], fc=NAV, ec=NAV, tc="white", sc="#cdd8e3")

# --- arrows + annotations in the gaps ---
arrow(y1 - h1 / 2, y2 + h2 / 2)
ops((y1 - h1 / 2 + y2 + h2 / 2) / 2,
    ["curate: source selection; window the series into",
     "fixed-length vectors; single-feature / permutation",
     "triviality and window-purity pre-filters"])

arrow(y2 - h2 / 2, y3 + h3 / 2)
gmid = (y2 - h2 / 2 + y3 + h3 / 2) / 2
ops(gmid,
    ["leak-free grouped split (unique vector + copies",
     "to one split); drop anomalies identical to a normal;",
     "deduplicate; severity OR-filter hardening;",
     "base-rate cap \u2264 0.5; train/val/test size gates"])
drop(gmid + 0.30, "\u22124,869 / 32 tasks")
drop(gmid - 0.75, "\u22121 task")

arrow(y3 - h3 / 2, y4 + h4 / 2)
gmid2 = (y3 - h3 / 2 + y4 + h4 / 2) / 2
ops(gmid2,
    ["score every detector; apply detector-based floors:",
     "separability   (oracle ap_norm > 0.05)",
     "minimum test anomalies   (\u2265 100)",
     "selection-trivial   (random-pick regret > 0.02)"])
drop(gmid2 + 0.15, "\u221217")
drop(gmid2 - 0.15, "\u22125")
drop(gmid2 - 0.45, "\u22122")

ax.set_title("The ADReal construction pipeline (one detector-free sequence)",
             fontsize=12.5, fontweight="bold", color=INK, y=0.99)
plt.tight_layout()
D = r"E:/tmp/claude/E--Projects-Submitted-ADRank/236d6247-42ad-4e62-9828-db625dfa055d/scratchpad/streamline"
fig.savefig(D + "/FIG_pipeline.png", dpi=125, facecolor="white", bbox_inches="tight")
print("saved")
