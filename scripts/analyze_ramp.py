"""Secondary analysis: does CrossQ dip DURING the fatigue ramp (0-400k steps)?

    python scripts/analyze_ramp.py results      # -> figures/ramp_crossq.*, tables/ramp_crossq.*

Pre-registered secondary metric (HYPOTHESIS.md, Amendment 1): the largest drop in return during
the ramp, relative to the Fixed run of the same seed. Each curve is the mean of the 10 evaluation
episodes per checkpoint, smoothed over 3 checkpoints (15k steps). "Largest drop" = largest fall
below the run's own running maximum within 0-400k. Fixed and No-fatigue runs show how large such
drops are from noise alone.

Caveat: before 400k, Drift is evaluated at its current (lower) fatigue level, so its raw return
is not comparable with Fixed; only drops within a run, and the final window, are.
"""
import glob
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
CONDS = [("fixed", "Fixed 0.5", "#2a78d6"), ("drift", "Drift 0 to 0.5", "#eb6834"),
         ("nofatigue", "No fatigue", "#1baf7a")]
RAMP_END, SMOOTH = 400_000, 3


def load():
    runs = {}
    for f in glob.glob(os.path.join(ROOT, "raw", "main_handreach_crossq_*_seed*_*", "evaluations.npz")):
        name = os.path.basename(os.path.dirname(f))
        cond, seed = name.split("_")[3], int(name.split("_")[4].replace("seed", ""))
        d = np.load(f)
        runs[(cond, seed)] = (d["timesteps"], d["results"].mean(1), d["successes"].mean(1))
    return runs


def smooth(y):
    return np.convolve(y, np.ones(SMOOTH) / SMOOTH, mode="valid"), SMOOTH // 2


def largest_drop(t, y):
    """Largest fall below the running maximum within the ramp; returns (drop, step of the low)."""
    ys, off = smooth(y)
    ts = t[off:off + len(ys)]
    m = ts <= RAMP_END
    ys, ts = ys[m], ts[m]
    gap = np.maximum.accumulate(ys) - ys
    i = int(gap.argmax())
    return float(gap[i]), int(ts[i])


def main():
    runs = load()
    seeds = sorted({s for (_, s) in runs})
    lines = ["| Seed | Largest drop: Fixed | Drift | No fatigue | Drift minus Fixed |",
             "|---|---|---|---|---|"]
    diffs = []
    for s in seeds:
        row, drops = [str(s)], {}
        for c, _, _ in CONDS:
            if (c, s) in runs:
                t, y, _ = runs[(c, s)]
                drops[c], at = largest_drop(t, y)
                row.append(f"{drops[c]:.0f} (at {at // 1000}k)")
            else:
                row.append("-")
        if "fixed" in drops and "drift" in drops:
            diffs.append(drops["drift"] - drops["fixed"])
            row.append(f"{diffs[-1]:+.0f}")
        lines.append("| " + " | ".join(row) + " |")
    rng = np.random.default_rng(0)
    d = np.array(diffs)
    boot = rng.choice(d, (10_000, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lines.append(f"\nMean Drift minus Fixed: {d.mean():+.1f}, 95% bootstrap CI [{lo:.1f}, {hi:.1f}] "
                 f"({len(d)} seeds). Positive = Drift dips more.")

    # Gap to No fatigue, per 100k window (shows how the cost of weakness builds up)
    lines.append("\nMean return per 100k window, seeds with all three conditions:")
    lines.append("| Window | " + " | ".join(lbl for _, lbl, _ in CONDS) + " | Drift level (mid-window) |")
    lines.append("|---|---|---|---|---|")
    full = [s for s in seeds if all((c, s) in runs for c, _, _ in CONDS)]
    for a in range(0, 500_000, 100_000):
        row = [f"{a // 1000}-{(a + 100_000) // 1000}k"]
        for c, _, _ in CONDS:
            v = []
            for s in full:
                t, y, _ = runs[(c, s)]
                v.append(y[(t > a) & (t <= a + 100_000)].mean())
            row.append(f"{np.mean(v):.0f}")
        mid = a + 50_000
        row.append(f"{0.5 * min(mid / RAMP_END, 1):.2f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append(f"(seeds {full})")

    text = "\n".join(lines)
    print(text)
    os.makedirs("tables", exist_ok=True)
    with open("tables/ramp_crossq.md", "w") as fh:
        fh.write(text + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(seeds), figsize=(7.2, 2.2), sharey=True)
    for ax, s in zip(axes, seeds):
        for c, lbl, col in CONDS:
            if (c, s) not in runs:
                continue
            t, y, _ = runs[(c, s)]
            ys, off = smooth(y)
            ax.plot(t[off:off + len(ys)] / 1000, ys, color=col, linewidth=1.5, label=lbl)
        ax.axvspan(0, RAMP_END / 1000, color="#e4e3df", alpha=0.35, linewidth=0)
        ax.set_title(f"Seed {s}", fontsize=8)
        ax.set_xticks([0, 200, 400])
        ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Evaluation return")
    fig.supxlabel("Environment steps (thousands); shaded = Drift ramp", fontsize=8, y=-0.08)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    fig.subplots_adjust(wspace=0.12)
    os.makedirs("figures", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"figures/ramp_crossq.{ext}", dpi=200, bbox_inches="tight")
    print("saved figures/ramp_crossq.png/.pdf, tables/ramp_crossq.md")


if __name__ == "__main__":
    main()
