"""Bridge check: Walker2d in the MyoSuite software setup (MuJoCo 3.1.2 + coursework patch) against the
reproduction (MuJoCo 2.3.7, unmodified code). One bridge seed per algorithm, so this is a check of the
CrossQ > SAC pattern and of the order of magnitude, not a statistical test.

    python scripts/analyze_bridge.py results      # -> tables/bridge_walker2d.md
"""
import glob
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
WINS = [(200_000, 250_000), (450_000, 500_000)]
lines = ["| Algorithm | Setup | Seeds | Mean return 200-250k | Mean return 450-500k | Minutes per run |",
         "|---|---|---|---|---|---|"]


def summary(pattern):
    vals, mins = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "raw", pattern, "evaluations.npz"))):
        d = np.load(f)
        t, r = d["timesteps"], d["results"].mean(1)
        vals.append([r[(t > a) & (t <= b)].mean() for a, b in WINS])
        for line in open(os.path.join(os.path.dirname(f), "meta.txt")):
            if line.startswith("wall_minutes:"):
                mins.append(float(line.split(":")[1]))
    return np.array(vals), mins


for algo, name in (("crossq", "CrossQ"), ("sac", "SAC")):
    for setup, pat in (("Reproduction (MuJoCo 2.3.7)", f"repro_walker2d_{algo}_seed*_*"),
                       ("Bridge (MuJoCo 3.1.2 + patch)", f"bridge_walker2d_{algo}_seed*_*")):
        v, m = summary(pat)
        if not len(v):
            continue
        cell = lambda x: f"{x.mean():.0f}" + (f" (range {x.min():.0f}-{x.max():.0f})" if len(x) > 1 else "")
        lines.append(f"| {name} | {setup} | {len(v)} | {cell(v[:, 0])} | {cell(v[:, 1])} | "
                     f"{min(m):.0f}-{max(m):.0f} |")
lines.append("\nOne evaluation episode every 5,000 steps (as in the reproduction); window means over evaluations.")
os.makedirs("tables", exist_ok=True)
open("tables/bridge_walker2d.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
