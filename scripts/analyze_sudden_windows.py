"""Exploratory comparisons around the sudden drop (paper Section V-D and Fig. 6).

    python scripts/analyze_sudden_windows.py [results_root]   # -> tables/sudden_windows.md

Paired by seed (the five seeds with a Sudden run, 11-15), 95% bootstrap CIs (10,000 resamples,
rng 0). Not pre-registered.
"""
import glob
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
SEEDS = [11, 12, 13, 14, 15]
WINDOWS = {"200k-250k (before the drop)": (200_000, 250_000), "255k (first evaluation after)": (250_000, 255_000),
           "255k-350k (H4 window)": (250_000, 350_000), "400k-500k (final)": (400_000, 500_000)}


def mean_return(cond, seed, a, b):
    f = glob.glob(os.path.join(ROOT, "raw", f"main_handreach_crossq_{cond}_seed{seed}_*", "evaluations.npz"))[0]
    d = np.load(f)
    t, r = d["timesteps"], d["results"].mean(1)
    return r[(t > a) & (t <= b)].mean()


def ci(x):
    b = np.random.default_rng(0).choice(x, (10_000, len(x))).mean(1)
    return np.percentile(b, [2.5, 97.5])


md = ["| Window | Comparison | Mean [95% bootstrap CI] | Per seed (11-15) |", "|---|---|---|---|"]
for name, (a, b) in WINDOWS.items():
    for x, y in (("fixed", "sudden"), ("nofatigue", "sudden")):
        d = np.array([mean_return(x, s, a, b) - mean_return(y, s, a, b) for s in SEEDS])
        lo, hi = ci(d)
        md.append(f"| {name} | {x} - {y} | {d.mean():.1f} [{lo:.1f}, {hi:.1f}] | {', '.join(f'{v:.0f}' for v in d)} |")
pre, post = WINDOWS["200k-250k (before the drop)"], WINDOWS["255k-350k (H4 window)"]
dd = np.array([(mean_return("fixed", s, *post) - mean_return("sudden", s, *post))
               - (mean_return("fixed", s, *pre) - mean_return("sudden", s, *pre)) for s in SEEDS])
lo, hi = ci(dd)
md.append(f"\nGap added by the drop (Fixed - Sudden over 255k-350k minus the same over 200k-250k): "
          f"{dd.mean():.1f} [{lo:.1f}, {hi:.1f}]. Exploratory, not pre-registered.")
os.makedirs("tables", exist_ok=True)
open("tables/sudden_windows.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
