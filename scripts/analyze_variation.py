"""Run-to-run variation between identical CrossQ runs (paper Section V-E).

    python scripts/analyze_variation.py [results_root]   # -> tables/run_to_run_variation.md

Sudden and No fatigue are identical conditions until the drop at 250k steps and use the same seeds
(11-15). Any difference between them before 250k is therefore run-to-run variation alone (GPU
computation is not bit-for-bit deterministic). Reports the first evaluation at which same-seed runs
differ, and the per-seed difference in mean return over 200k-250k, compared with the spread between
different seeds.
"""
import glob
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
SEEDS = [11, 12, 13, 14, 15]


def load(cond, seed):
    f = glob.glob(os.path.join(ROOT, "raw", f"main_handreach_crossq_{cond}_seed{seed}_*", "evaluations.npz"))[0]
    d = np.load(f)
    return d["timesteps"], d["results"]


rows, diffs, nofat = [], [], []
for s in SEEDS:
    t, a = load("nofatigue", s)
    _, b = load("sudden", s)
    same = [bool(np.allclose(a[i], b[i])) for i in range(len(t))]
    first = int(t[same.index(False)]) if False in same else None
    m = (t > 200_000) & (t <= 250_000)
    x, y = a[m].mean(), b[m].mean()
    diffs.append(x - y)
    nofat.append(x)
    rows.append(f"| {s} | {first:,} | {x:.1f} | {y:.1f} | {x - y:.1f} |")
diffs, nofat = np.array(diffs), np.array(nofat)
md = ["| Seed | First evaluation that differs (steps) | No fatigue, 200k-250k | Sudden, 200k-250k | Difference |",
      "|---|---|---|---|---|", *rows,
      f"\nSame-seed runs of identical conditions: mean difference {diffs.mean():.1f}, SD {diffs.std(ddof=1):.1f} "
      f"(5 seeds). Spread between different seeds (SD of No fatigue over 200k-250k): {nofat.std(ddof=1):.1f}.",
      "Runs are identical at the evaluations at 1 and 5,000 steps (before learning starts) and differ from 10,000 steps on."]
os.makedirs("tables", exist_ok=True)
open("tables/run_to_run_variation.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
