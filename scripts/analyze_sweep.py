"""Amendment 4 (exploratory): final policies evaluated at fixed fatigue levels.

    python scripts/analyze_sweep.py [results_root]   # -> figures/fatigue_sweep.*, tables/sweep_crossq.md

Reads results/raw/sweep_handreach_crossq_<condition>_seed<k>_*/fatigue_sweep.npz (written by
src/fatigue_sweep.py) and reports the quantities fixed in Amendment 4:
  E1  return of No-fatigue-trained policies against test fatigue level (physical cost)
  E2  at test level 0.5: Fixed-trained minus No-fatigue-trained return, paired by seed
  E3  at test level 0:   No-fatigue-trained minus Fixed-trained return, paired by seed
  E4  per training condition, the lowest test level at which the mean solved fraction falls below
      half of its value at level 0 (descriptive)
All quantities are exploratory and reported per seed, with 95% bootstrap CIs and t-intervals over seeds.
"""
import glob
import os
import re
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
CONDS = [("nofatigue", "Trained without fatigue", "#1baf7a"), ("fixed", "Trained at fatigue 0.5", "#2a78d6")]
INK, MUTED, GRID = "#2b2a28", "#6b6a66", "#e4e3df"
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
                     "text.color": INK, "pdf.fonttype": 42})
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)


def boot_ci(x, n=10_000):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    b = np.random.default_rng(0).choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def t_ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * np.std(x, ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def fmt(m, ci):
    return f"{m:.1f}" if np.isnan(ci[0]) else f"{m:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]"


data = {}   # cond -> {seed: (levels, returns(L,N), solved(L,N))}
for f in glob.glob(os.path.join(ROOT, "raw", "sweep_handreach_crossq_*_seed*_*", "fatigue_sweep.npz")):
    m = re.match(r"sweep_handreach_crossq_(\w+?)_seed(\d+)_", os.path.basename(os.path.dirname(f)))
    d = np.load(f)
    data.setdefault(m.group(1), {})[int(m.group(2))] = (d["levels"], d["returns"], d["solved"])
if not data:
    sys.exit("no fatigue_sweep.npz files found")

levels = next(iter(next(iter(data.values())).values()))[0]
per_seed = {c: {s: (v[1].mean(1), v[2].mean(1)) for s, v in d.items()} for c, d in data.items()}

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharex=True)
for c, lab, col in CONDS:
    if c not in per_seed:
        continue
    R = np.array([v[0] for v in per_seed[c].values()])
    S = np.array([v[1] for v in per_seed[c].values()])
    for ax, Y in ((axes[0], R), (axes[1], S)):
        for y in Y:          # with 3 seeds, show every seed rather than a bootstrap band
            ax.plot(levels, y, color=col, linewidth=0.6, alpha=0.35)
        ax.plot(levels, Y.mean(0), color=col, linewidth=1.5, marker="o", markersize=2.5,
                label=f"{lab} (n = {len(Y)})")
for ax, ylab in ((axes[0], "Return"), (axes[1], "Solved fraction")):
    ax.axvline(0.5, color=MUTED, linestyle=(0, (2, 2)), linewidth=0.8)
    ax.set_xlabel("Test fatigue level")
    ax.set_ylabel(ylab)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[1].set_ylim(0, 1)
h, lb = axes[0].get_legend_handles_labels()
fig.legend(h, lb, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=1, frameon=False, fontsize=7)
fig.subplots_adjust(wspace=0.45)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/fatigue_sweep.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- table
md = ["| Training condition | Seeds | " + " | ".join(f"{l:g}" for l in levels) + " |",
      "|---|---|" + "---|" * len(levels)]
for c, lab, _ in CONDS:
    if c in per_seed:
        R = np.array([v[0] for v in per_seed[c].values()])
        S = np.array([v[1] for v in per_seed[c].values()])
        md.append(f"| {lab}, return | {len(R)} | " + " | ".join(f"{x:.0f}" for x in R.mean(0)) + " |")
        md.append(f"| {lab}, solved | {len(S)} | " + " | ".join(f"{x:.2f}" for x in S.mean(0)) + " |")
md.append("\nMean over seeds of each policy's mean over 50 episodes per level (same targets at every level).")


def paired(a, b, lvl):
    i = int(np.argmin(np.abs(levels - lvl)))
    ss = sorted(set(per_seed.get(a, {})) & set(per_seed.get(b, {})))
    return ss, np.array([per_seed[a][s][0][i] - per_seed[b][s][0][i] for s in ss])


md.append("\n| Quantity | Seeds | Per seed | Mean [95% bootstrap CI] | 95% t-interval |\n|---|---|---|---|---|")
if "nofatigue" in per_seed:      # E1 at the trained-weak level: same policy, same targets, level 0 minus 0.5
    i0, i5 = int(np.argmin(np.abs(levels))), int(np.argmin(np.abs(levels - 0.5)))
    ss = sorted(per_seed["nofatigue"])
    d = np.array([per_seed["nofatigue"][k][0][i0] - per_seed["nofatigue"][k][0][i5] for k in ss])
    tc = t_ci(d)
    md.append(f"| E1 at 0.5: No-fatigue-trained, test level 0 minus 0.5 (physical cost) | {len(d)} | "
              + ", ".join(str(int(round(x))) or "0" for x in d) + f" | {fmt(d.mean(), boot_ci(d))} | [{tc[0]:.1f}, {tc[1]:.1f}] |")
for name, a, b, lvl in (("E2: Fixed-trained minus No-fatigue-trained, test level 0.5", "fixed", "nofatigue", 0.5),
                        ("E3: No-fatigue-trained minus Fixed-trained, test level 0", "nofatigue", "fixed", 0.0)):
    ss, d = paired(a, b, lvl)
    if len(d):
        tc = t_ci(d)
        md.append(f"| {name} | {len(d)} | " + ", ".join(str(int(round(x))) or "0" for x in d) + f" | {fmt(d.mean(), boot_ci(d))} | "
                  + ("-" if np.isnan(tc[0]) else f"[{tc[0]:.1f}, {tc[1]:.1f}]") + " |")
if "nofatigue" in per_seed:
    R = np.array([v[0] for v in per_seed["nofatigue"].values()])
    md.append(f"\nE1: No-fatigue-trained return falls from {R.mean(0)[0]:.0f} at level 0 to "
              f"{R.mean(0)[-1]:.0f} at level {levels[-1]:g}.")
for c, lab, _ in CONDS:
    if c in per_seed:
        S = np.array([v[1] for v in per_seed[c].values()]).mean(0)
        below = [l for l, s in zip(levels, S) if s < 0.5 * S[0]]
        md.append(f"E4 ({lab}): " + (f"solved fraction first below half its level-0 value at {below[0]:g}."
                                     if below else "solved fraction never falls below half its level-0 value."))
md.append("\nAll quantities are exploratory (Amendment 4). With 3 seeds the bootstrap CI is unreliable, so the t-interval is the main interval.")
open("tables/sweep_crossq.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
