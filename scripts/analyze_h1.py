"""Primary pre-registered test H1 (HYPOTHESIS.md): CrossQ, Fixed vs Drift, final 100k steps.

    python scripts/analyze_h1.py results      # -> tables/h1_crossq.md

Per run: mean evaluation return over the 20 evaluations after 400k steps (405k-500k, 200 episodes).
Per seed: d = Fixed - Drift. H1 supported if the 95% bootstrap CI of mean d (10,000 resamples of
seeds) lies entirely above 0; contradicted if entirely below 0; otherwise inconclusive.

Minimum detectable effect (Amendment 2): the smallest true mean difference that a paired two-sided
t-test at alpha = 0.05 would detect with 80% power, given the observed SD of d over n seeds:
MDE = (t_{0.975, n-1} + t_{0.80, n-1}) * SD(d) / sqrt(n).
"""
import glob
import os
import sys

import numpy as np

ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"


def final_window(cond):
    out = {}
    for f in glob.glob(os.path.join(ROOT, "raw", f"main_handreach_crossq_{cond}_seed*_*", "evaluations.npz")):
        seed = int(os.path.basename(os.path.dirname(f)).split("_")[4].replace("seed", ""))
        d = np.load(f)
        m = d["timesteps"] > 400_000
        assert m.sum() == 20, f"{f}: expected 20 evaluations after 400k, found {m.sum()}"
        out[seed] = (d["results"][m].mean(), d["successes"][m].mean())
    return out


fixed, drift, nofat = final_window("fixed"), final_window("drift"), final_window("nofatigue")
seeds = sorted(set(fixed) & set(drift))
lines = ["| Seed | No fatigue | Fixed | Drift | d = Fixed - Drift | Solved rate Fixed / Drift |",
         "|---|---|---|---|---|---|"]
for s in seeds:
    nf = f"{nofat[s][0]:.0f}" if s in nofat else "-"
    lines.append(f"| {s} | {nf} | {fixed[s][0]:.0f} | {drift[s][0]:.0f} | {fixed[s][0] - drift[s][0]:+.0f} "
                 f"| {fixed[s][1]:.2f} / {drift[s][1]:.2f} |")
d = np.array([fixed[s][0] - drift[s][0] for s in seeds])
boot = np.random.default_rng(0).choice(d, (10_000, len(d))).mean(1)
lo, hi = np.percentile(boot, [2.5, 97.5])
from scipy import stats  # noqa: E402
n = len(d)
mde = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * d.std(ddof=1) / np.sqrt(n)
verdict = "SUPPORTED" if lo > 0 else "CONTRADICTED" if hi < 0 else "INCONCLUSIVE"
lines.append(f"\nMean d = {d.mean():+.1f}, 95% bootstrap CI [{lo:.1f}, {hi:.1f}], {len(d)} seeds: H1 {verdict}.")
lines.append(f"SD of d = {d.std(ddof=1):.1f}. Minimum detectable effect (80% power, alpha 0.05, paired t): {mde:.0f}.")
text = "\n".join(lines)
print(text)
os.makedirs("tables", exist_ok=True)
with open("tables/h1_crossq.md", "w") as fh:
    fh.write(text + "\n")
