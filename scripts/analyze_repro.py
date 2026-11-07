"""Reproduction analysis: CrossQ vs SAC on Walker2d, aggregated as in Bhatt et al. (2024), Fig. 5.

    python scripts/analyze_repro.py [results_dir=results]

Reads   results/raw/repro_walker2d_*/evaluations.npz  and  results/random/Walker2d-v4.npz
Writes  figures/repro_walker2d.pdf (+ .png)  and  tables/repro_walker2d.csv (+ .md)

Aggregation (matches the paper): interquartile mean (IQM) over seeds, with a 70% quantile
interval (15th-85th percentile over seeds). Each evaluation is a single episode (as in the
paper), so each seed's curve is first smoothed with a trailing mean over the last 5
evaluations (25,000 environment steps). The table additionally gives a 95% bootstrap
confidence interval for the IQM (resampling seeds, 10,000 resamples, fixed RNG seed).
"""
import csv
import glob
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR, TAB_DIR = os.path.join(KIT, "figures"), os.path.join(KIT, "tables")
SMOOTH = 5                                  # evaluations in the trailing mean (5 x 5,000 steps)
CHECKPOINTS = [100_000, 250_000, 500_000]
ALGOS = [("crossq", "CrossQ", "#2a78d6"), ("sac", "SAC", "#eb6834")]
# Approximate values read by eye from the paper's Fig. 5 (Walker2d panel) at 0.5M steps.
PAPER_AT_500K = {"crossq": "~6300", "sac": "~4000"}


def iqm(x, axis=0):
    """Interquartile mean: mean of the values between the 25th and 75th percentile."""
    x = np.sort(np.asarray(x), axis=axis)
    n = x.shape[axis]
    lo, hi = int(np.floor(0.25 * n)), int(np.ceil(0.75 * n))
    return np.take(x, range(lo, hi), axis=axis).mean(axis=axis)


def bootstrap_ci(values, stat=iqm, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values)
    stats = [stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n)]
    return np.percentile(stats, [2.5, 97.5])


def trailing_mean(y, w):
    c = np.cumsum(np.insert(y, 0, 0.0))
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def load(algo):
    runs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "raw", f"repro_walker2d_{algo}_seed*", "evaluations.npz"))):
        seed = int(re.search(r"_seed(\d+)_", path).group(1))
        if seed in runs:
            raise SystemExit(f"Two runs for {algo} seed {seed}: decide which one counts first.")
        d = np.load(path)
        runs[seed] = (d["timesteps"], d["results"].mean(axis=1))
    if not runs:
        raise SystemExit(f"No runs found for {algo} under {ROOT}/raw")
    steps = runs[min(runs)][0]
    for s, (t, _) in runs.items():
        if not np.array_equal(t, steps):
            raise SystemExit(f"{algo} seed {s} was evaluated at different steps; cannot aggregate.")
    raw = np.stack([runs[s][1] for s in sorted(runs)])
    smooth = np.stack([trailing_mean(r, SMOOTH) for r in raw])
    return sorted(runs), steps, raw, smooth


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    rnd = np.load(os.path.join(ROOT, "random", "Walker2d-v4.npz"))["returns"]
    rnd_mean = float(rnd.mean())

    data = {a: load(a) for a, _, _ in ALGOS}

    # ---------- table ----------
    rows = []
    for a, label, _ in ALGOS:
        seeds, steps, _, smooth = data[a]
        for cp in CHECKPOINTS:
            i = int(np.argmin(np.abs(steps - cp)))
            vals = smooth[:, i]
            lo70, hi70 = np.percentile(vals, [15, 85])
            ci_lo, ci_hi = bootstrap_ci(vals)
            rows.append({
                "algorithm": label, "env_steps": int(steps[i]), "n_seeds": len(seeds),
                "iqm": round(float(iqm(vals)), 1),
                "q15": round(float(lo70), 1), "q85": round(float(hi70), 1),
                "iqm_ci95_lo": round(float(ci_lo), 1), "iqm_ci95_hi": round(float(ci_hi), 1),
                "per_seed": " ".join(f"{v:.0f}" for v in vals),
                "paper_fig5_approx": PAPER_AT_500K[a] if cp == 500_000 else "",
            })
    with open(os.path.join(TAB_DIR, "repro_walker2d.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(TAB_DIR, "repro_walker2d.md"), "w") as f:
        f.write(f"Random policy (100 episodes): mean return {rnd_mean:.1f}\n\n")
        f.write("| Algorithm | Steps | IQM | 70% interval | IQM 95% CI | Per seed | Paper Fig. 5 (approx.) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['algorithm']} | {r['env_steps']:,} | {r['iqm']:.0f} | {r['q15']:.0f}-{r['q85']:.0f} | "
                    f"{r['iqm_ci95_lo']:.0f}-{r['iqm_ci95_hi']:.0f} | {r['per_seed']} | {r['paper_fig5_approx']} |\n")

    # ---------- figure ----------
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for a, label, color in ALGOS:
        seeds, steps, _, smooth = data[a]
        x = steps / 1e6
        ax.fill_between(x, np.percentile(smooth, 15, axis=0), np.percentile(smooth, 85, axis=0),
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(x, iqm(smooth, axis=0), color=color, linewidth=2, label=f"{label} ({len(seeds)} seeds)")
        ax.annotate(label, (x[-1], iqm(smooth, axis=0)[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", color="#0b0b0b", fontsize=8)
    ax.axhline(rnd_mean, color="#52514e", linestyle="--", linewidth=1, label="Random policy")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Return (IQM over seeds)")
    ax.set_xlim(0, 0.5)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=7, handlelength=1.6, columnspacing=1.0)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"repro_walker2d.{ext}"), dpi=300, bbox_inches="tight")
    print(open(os.path.join(TAB_DIR, "repro_walker2d.md")).read())


if __name__ == "__main__":
    main()
