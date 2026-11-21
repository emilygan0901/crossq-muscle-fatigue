"""Schematic of the training setup (paper figure).

    python scripts/make_schematic.py      # -> figures/setup_schematic.png/.pdf

Shows how the fatigue schedule, the MyoSuite task, the replay buffer and CrossQ's batch-normalised
networks fit together. Drawn with matplotlib so it can be regenerated and exported as vector.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED, LINE = "#2b2a28", "#6b6a66", "#8a8984"
FILL = {"ours": "#fde7d9", "task": "#e3eefa", "learner": "#e4f5ed", "data": "#f1f0ec"}
EDGE = {"ours": "#eb6834", "task": "#2a78d6", "learner": "#1baf7a", "data": "#8a8984"}
plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

fig, ax = plt.subplots(figsize=(3.4, 2.75))
ax.set_xlim(0, 100)
ax.set_ylim(0, 81)
ax.axis("off")


def box(x, y, w, h, kind, title, sub):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
                                fc=FILL[kind], ec=EDGE[kind], lw=1.0))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=6.6, color=INK, weight="bold")
    ax.text(x + w / 2, y + h - 8.6, sub, ha="center", va="top", fontsize=5.4, color=MUTED, linespacing=1.25)


def arrow(p, q, text=None, tx=0, ty=0, rad=0.0, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.9, color=LINE,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + tx, (p[1] + q[1]) / 2 + ty, text, ha=ha, va="center",
                fontsize=5.8, color=MUTED, linespacing=1.2)


# row 1: our fatigue schedule (top), evaluation (right)
box(2, 61, 44, 18, "ours", "Fatigue schedule  ℓ(t)", "this work; t = training steps\nFixed, Drift or Sudden")
box(56, 61, 42, 18, "data", "Evaluation", "10 episodes every 5k steps\nat the current level ℓ(t)")
# row 2: task (left), agent (right)
box(2, 31, 44, 20, "task", "MyoSuite Hand Reach", "39 muscles, 5 targets\nMF = ℓ(t) at each reset")
box(56, 31, 42, 20, "learner", "Actor  π(a | s)", "BRN layers (CrossQ)\noutputs muscle commands")
# row 3: replay buffer (left), critic (right)
box(2, 2, 44, 19, "data", "Replay buffer", "1M: holds the whole run\nuniform batches of 256")
box(56, 2, 42, 19, "learner", "Critic  Q(s, a)", "BRN, joint pass over\n(s,a) and (s′,a′)")

arrow((24, 61), (24, 51.6), "ℓ(t) at\nepisode start", tx=-1.2, ha="right")
arrow((46, 70), (56, 70), "ℓ(t)", ty=2.4)
arrow((56, 41), (46, 41), "aₜ", ty=2.4)
arrow((46, 36), (56, 36), "sₜ", ty=-2.4)
arrow((24, 31), (24, 21.6), "(s, a, r, s′)", tx=-1.2, ha="right")
arrow((46, 11.5), (56, 11.5), "batch", ty=2.4)
arrow((77, 21.6), (77, 31), "policy\nupdate", tx=1.2, ha="left")

os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/setup_schematic.{ext}", dpi=200, bbox_inches="tight")
print("saved figures/setup_schematic.png/.pdf")
