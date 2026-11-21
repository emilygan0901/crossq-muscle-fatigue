"""Three-panel schematic of the method (paper figure).

    python scripts/make_setup_figure.py      # -> figures/setup_figure.png/.pdf

A: RL loop with CrossQ (batch-normalised actor and critic, replay buffer).
B: inside the environment: our fatigue schedule sets the muscles' fatigued fraction at each reset.
C: one training run (Drift shown): episodes at a rising fatigue level, the replay buffer holding a
   mixture of all levels, and evaluation every 5,000 steps.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle  # noqa: E402

plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
                     "mathtext.fontset": "dejavusans"})
INK, MUTED, LINE = "#2b2a28", "#6b6a66", "#7d7c77"
AGENT, ENV, OURS, DATA = "#1baf7a", "#2a78d6", "#eb6834", "#8a8984"
AGENT_F, ENV_F, OURS_F = "#e8f6ef", "#e8f1fb", "#fdebe1"

fig = plt.figure(figsize=(3.45, 5.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 150)
ax.axis("off")


def box(x, y, w, h, ec, fc="white", lw=1.0, ls="-", r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls))


def arrow(p, q, rad=0.0, ls="-", color=LINE, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6.5, lw=lw, color=color,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def poly(pts, color=LINE, lw=0.9, head=True):
    """Right-angled connector through pts, arrowhead at the end."""
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="butt")
    if head:
        arrow(pts[-2], pts[-1], color=color, lw=lw)


def txt(x, y, s, size=7, color=INK, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, **kw)


def label(x, y, s):
    txt(x, y, s, size=10, weight="bold", ha="left")


def bars(x, y, heights, w=1.6, gap=0.9, color=AGENT):
    for i, h in enumerate(heights):
        ax.add_patch(FancyBboxPatch((x + i * (w + gap), y), w, h, boxstyle="round,pad=0,rounding_size=0.5",
                                    fc=AGENT_F, ec=color, lw=0.7))


def net(cx, cy, scale=1.0, color=AGENT):
    layers = [3, 4, 4, 2]
    xs = [cx - 5 * scale + i * 3.4 * scale for i in range(len(layers))]
    pts = [[(x, cy + (j - (n - 1) / 2) * 2.0 * scale) for j in range(n)] for x, n in zip(xs, layers)]
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=0.25, alpha=0.55)
    for layer in pts:
        for p in layer:
            ax.add_patch(Circle(p, 0.75 * scale, fc="white", ec=color, lw=0.6))


# ============================================================== A: RL loop
label(1, 146, "A")
# agent
box(13, 112, 40, 31, AGENT, AGENT_F)
txt(33, 140, r"Agent (CrossQ)", 7.2, AGENT, weight="bold")
net(24, 127, 0.95)
txt(41, 131.5, r"actor $\pi(a\,|\,s)$", 6.3)
txt(41, 125.5, r"critic $Q(s,a)$", 6.3)
txt(41, 119.5, "batch renorm.,", 5.6, MUTED)
txt(41, 116, "no target net", 5.6, MUTED)
# muscle activations
txt(76, 140.5, r"$a_t \sim \pi(\cdot\,|\,s_t)$", 7)
bars(67, 127, [7, 10, 5, 8, 4, 9], w=1.6, gap=1.1)
txt(74.5, 123.5, "39 muscle", 5.8, AGENT)
txt(74.5, 120.3, "activations", 5.8, AGENT)
# environment
box(62, 101, 31, 12, ENV, ENV_F)
txt(77.5, 107, "Environment", 7.2, ENV, weight="bold")
txt(77.5, 103.3, "(panel B)", 5.6, MUTED)
# replay buffer
box(13, 101, 40, 7.5, DATA, "#f3f2ee")
txt(33, 104.75, "Replay buffer (1M, whole run)", 6.0)
# arrows
poly([(53, 136), (96, 136), (96, 113.2)])                     # a_t -> env
txt(95, 117.5, r"$a_t$", 7, ha="right")
arrow((62, 104.75), (53.2, 104.75))                           # env -> buffer
txt(57.6, 98.7, r"$(s_t, a_t, r_{t+1}, s_{t+1})$", 5.8)
poly([(8, 104.75), (13, 104.75)], head=False)
poly([(13, 104.75), (8, 104.75), (8, 127), (12.8, 127)])      # buffer -> agent
txt(7.2, 116, "uniform batch\nof 256", 5.6, MUTED, ha="right", rotation=90)
poly([(65, 113), (65, 118), (53.2, 118)], color=ENV)          # s_t to actor
txt(59, 120.3, r"$s_t$", 6.6, ENV)

# ============================================================== B: inside the environment
label(1, 96, "B")
box(9, 52, 87, 43, ENV, "white", lw=1.1)
txt(52.5, 92, "Environment: MyoSuite Hand Reach", 7.2, ENV, weight="bold")
# fatigue schedule (ours)
box(12, 71, 28, 16, OURS, OURS_F)
txt(26, 84.2, "Fatigue schedule", 6.5, OURS, weight="bold")
txt(26, 80.3, r"$\ell(t)$", 7, OURS)
txt(26, 76.9, r"$t$ = training step", 5.6)
txt(26, 73.6, "this work", 5.6, OURS, style="italic")
# dynamics
box(53, 69, 40, 18, ENV, "white")
txt(73, 84, "Dynamics", 7, ENV, weight="bold")
txt(73, 79.3, "MuJoCo + 39-muscle hand", 5.8)
txt(73, 75.6, "fatigue: MA + MR + MF = 1", 5.8)
txt(73, 71.9, r"activation $\leq 1-$MF", 5.8)
# reward and state
box(12, 55, 28, 11, ENV, "white")
txt(26, 62.5, "Reward", 6.8, ENV, weight="bold")
txt(26, 58.3, r"$-d_t$ + bonus $-$ penalty", 5.6)
box(53, 55, 40, 11, ENV, "white")
txt(73, 62.5, "State", 6.8, ENV, weight="bold")
txt(73, 58.3, "joints, fingertips, errors, activations", 5.4)
# arrows in B
arrow((40, 79.5), (52.8, 79.5), color=OURS, lw=1.1)
txt(46.4, 82.4, r"MF$=\ell(t)$", 5.6, OURS)
txt(46.4, 76.8, "at reset", 5.4, OURS)
poly([(99, 69), (99, 78), (93.2, 78)])                        # a_t into dynamics
txt(99.4, 66.5, r"$a_t$", 7, ha="right")
poly([(73, 69), (73, 66.2)])                                  # dynamics -> state
poly([(57, 69), (57, 67.5), (34, 67.5), (34, 66.2)])          # dynamics -> reward
poly([(12, 60.5), (5, 60.5)])                                 # reward out
txt(1.2, 63.4, r"$r_{t+1}$", 7, ha="left")
poly([(72, 55), (72, 50.8)])                                  # state out
txt(73, 50.2, r"$s_{t+1}$", 7, ha="left", va="top")

# ============================================================== C: one training run (Drift)
label(1, 45, "C")
txt(58, 44, "One training run (Drift shown)", 7.2, INK, weight="bold")
X0, X1 = 20, 96


def xs(step):  # training step (thousands) -> x
    return X0 + (X1 - X0) * step / 500


def lvl(step):  # Drift schedule
    return 0.5 * min(step / 400, 1.0)


def shade(level):  # fatigue 0..0.5 -> light to dark orange
    import matplotlib.colors as mc
    return mc.to_hex([(1 - level / 0.5) * a + (level / 0.5) * b
                      for a, b in zip(mc.to_rgb("#fdf3ec"), mc.to_rgb(OURS))])


# episodes strip
txt(X0 - 1, 36.5, "episodes", 5.8, MUTED, ha="right")
for k in range(25):
    s0, s1 = k * 20, k * 20 + 16
    ax.add_patch(Rectangle((xs(s0), 34), xs(s1) - xs(s0), 5, fc=shade(lvl(s0)), ec=OURS, lw=0.35))
# replay buffer strip: mixture grows with training
txt(X0 - 1, 25.5, "buffer", 5.8, MUTED, ha="right")
n = 50
for i in range(n):
    s0 = i * 500 / n
    ax.add_patch(Rectangle((xs(s0), 23), xs(s0 + 500 / n) - xs(s0), 5, fc=shade(lvl(s0)), ec="none"))
ax.add_patch(Rectangle((xs(0), 23), xs(500) - xs(0), 5, fc="none", ec=DATA, lw=0.6))
txt(xs(250), 20.2, "by 500k the buffer holds transitions from every fatigue level", 5.4, MUTED)
# evaluation ticks
txt(X0 - 1, 14.5, "evaluation", 5.8, MUTED, ha="right")
for k in range(0, 501, 25):
    ax.plot([xs(k), xs(k)], [12.5, 16.5], color=AGENT, lw=0.6)
txt(xs(250), 9.6, r"every 5k steps (25k drawn): 10 episodes at the current $\ell(t)$", 5.4, MUTED)
# axis
ax.plot([X0, X1], [5.5, 5.5], color=INK, lw=0.7)
for k in (0, 100, 200, 300, 400, 500):
    ax.plot([xs(k), xs(k)], [5.5, 4.4], color=INK, lw=0.6)
    txt(xs(k), 2.4, f"{k}k", 5.8)
txt(X0 - 2, 5.5, "steps", 5.8, MUTED, ha="right")
# level legend
for i, v in enumerate([0, 0.25, 0.5]):
    ax.add_patch(Rectangle((60 + i * 12, 40.4), 3, 2.2, fc=shade(v), ec=OURS, lw=0.35))
    txt(64.2 + i * 12, 41.5, f"{v:g}", 5.6, MUTED, ha="left")
txt(58.5, 41.5, r"$\ell$:", 5.8, MUTED, ha="right")

os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/setup_figure.{ext}", dpi=220)
print("saved figures/setup_figure.png/.pdf")
