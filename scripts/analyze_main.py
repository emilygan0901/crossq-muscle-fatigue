"""Main-experiment figures and tables (Hand Reach). Re-run whenever new runs are added.

    python scripts/analyze_main.py results

Writes
  figures/schedules.*         fatigue level against training steps for each condition
  figures/main_curves.*       learning curves, CrossQ and SAC, mean and 95% bootstrap band over seeds
  figures/main_differences.*  per-seed paired differences, mean, 95% CI and minimum detectable effect
  tables/main_results.md/.csv final return, solved fraction, AUC per algorithm and condition
  tables/hypotheses.md        every pre-registered comparison: estimate, CI, MDE, verdict
  tables/settings_compute.md  settings, software versions, hardware and compute per group of runs

Definitions (HYPOTHESIS.md): final window = 20 evaluations after 400k steps; Sudden window = 20
evaluations at 255k-350k; AUC = mean evaluation return over all evaluations (0-500k), i.e. area
under the learning curve divided by its length. Intervals: 95% bootstrap over seeds (10,000
resamples, RNG seed 0). MDE: smallest true mean difference detectable with 80% power by a two-sided
paired t-test at alpha 0.05, from the observed SD of the per-seed differences.
"""
import csv
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
ALGOS = [("crossq", "CrossQ"), ("sac", "SAC")]
CONDS = [("nofatigue", "No fatigue", "#1baf7a"), ("fixed", "Fixed 0.5", "#2a78d6"),
         ("drift", "Drift 0 to 0.5", "#eb6834"), ("sudden", "Sudden at 250k", "#eda100")]
LABEL = {c: l for c, l, _ in CONDS}
COLOR = {c: k for c, _, k in CONDS}
FINAL, SUDDEN = (400_000, 500_000), (250_000, 350_000)
INK, MUTED, GRID = "#2b2a28", "#6b6a66", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42})
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)


# ---------------------------------------------------------------- data
def load():
    runs = {}
    for f in glob.glob(os.path.join(ROOT, "raw", "main_handreach_*_seed*_*", "evaluations.npz")):
        run = os.path.basename(os.path.dirname(f))
        m = re.match(r"main_handreach_(\w+?)_(\w+?)_seed(\d+)_", run)
        algo, cond, seed = m.group(1), m.group(2), int(m.group(3))
        d = np.load(f)
        runs[(algo, cond, seed)] = dict(t=d["timesteps"], r=d["results"].mean(1),
                                        s=d["successes"].mean(1), dir=os.path.dirname(f))
    return runs


def window(run, w, key="r"):
    m = (run["t"] > w[0]) & (run["t"] <= w[1])
    return run[key][m].mean()


def boot_ci(x, n=10_000):
    x = np.asarray(x, float)
    if len(x) < 3:
        return np.nan, np.nan
    b = np.random.default_rng(0).choice(x, (n, len(x))).mean(1)
    return tuple(np.percentile(b, [2.5, 97.5]))


def iqm(x):
    x = np.sort(np.asarray(x, float))
    k = int(np.floor(0.25 * len(x)))
    return x[k:len(x) - k].mean() if len(x) >= 4 else x.mean()


def mde(d):
    n = len(d)
    if n < 3:
        return np.nan
    return (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * np.std(d, ddof=1) / np.sqrt(n)


def seeds(runs, algo, cond):
    return sorted(s for (a, c, s) in runs if a == algo and c == cond)


runs = load()
random_ref = {}
for lvl in ("0", "0.5"):
    f = os.path.join(ROOT, "random", f"myoFatiHandReachRandomDrift-v0_fatigue{lvl}.npz")
    if os.path.exists(f):
        random_ref[lvl] = float(np.load(f)["returns"].mean())


def fmt(m, ci, nd=0):
    if np.isnan(ci[0]):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


# ---------------------------------------------------------------- Table 1: results
rows, md = [], ["| Algorithm | Condition | Seeds | Final return | IQM | Solved fraction | AUC (mean return 0-500k) |",
                "|---|---|---|---|---|---|---|"]
for a, al in ALGOS:
    for c, cl, _ in CONDS:
        ss = seeds(runs, a, c)
        if not ss:
            continue
        fin = [window(runs[(a, c, s)], FINAL) for s in ss]
        sol = [window(runs[(a, c, s)], FINAL, "s") for s in ss]
        auc = [runs[(a, c, s)]["r"].mean() for s in ss]
        rows.append(dict(algorithm=al, condition=cl, n=len(ss), seeds=" ".join(map(str, ss)),
                         final_mean=np.mean(fin), final_lo=boot_ci(fin)[0], final_hi=boot_ci(fin)[1],
                         final_iqm=iqm(fin), solved_mean=np.mean(sol), solved_lo=boot_ci(sol)[0],
                         solved_hi=boot_ci(sol)[1], auc_mean=np.mean(auc), auc_lo=boot_ci(auc)[0],
                         auc_hi=boot_ci(auc)[1]))
        md.append(f"| {al} | {cl} | {len(ss)} | {fmt(np.mean(fin), boot_ci(fin))} | {iqm(fin):.0f} | "
                  f"{fmt(np.mean(sol), boot_ci(sol), 2)} | {fmt(np.mean(auc), boot_ci(auc))} |")
md.append("\nMean [95% bootstrap CI over seeds]; no CI with fewer than 3 seeds. Final = 20 evaluations "
          "(200 episodes) after 400k steps. Solved = MyoSuite 'solved' at episode end. IQM over seeds "
          "(equals the mean below 4 seeds).")
if random_ref:
    md.append("Random policy: " + ", ".join(f"{v:.1f} at fatigue {k}" for k, v in random_ref.items()) + ".")
open("tables/main_results.md", "w").write("\n".join(md) + "\n")
with open("tables/main_results.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


# ---------------------------------------------------------------- Table 2: hypotheses
def paired(algo, a_cond, b_cond, win):
    ss = sorted(set(seeds(runs, algo, a_cond)) & set(seeds(runs, algo, b_cond)))
    d = np.array([window(runs[(algo, a_cond, s)], win) - window(runs[(algo, b_cond, s)], win) for s in ss])
    return ss, d


def verdict(ci, n):
    if n < 3:
        return "too few seeds"
    return "supported" if ci[0] > 0 else "contradicted" if ci[1] < 0 else "inconclusive"


CONTRASTS = [  # (id, label, algo, a, b, window)
    ("Check", "Cost of fatigue (No fatigue - Fixed)", "crossq", "nofatigue", "fixed", FINAL),
    ("H1", "Drift penalty (Fixed - Drift)", "crossq", "fixed", "drift", FINAL),
    ("H4", "Sudden penalty (Fixed - Sudden, 255k-350k)", "crossq", "fixed", "sudden", SUDDEN),
    ("Check", "Cost of fatigue (No fatigue - Fixed)", "sac", "nofatigue", "fixed", FINAL),
    ("H2 part", "Drift penalty (Fixed - Drift)", "sac", "fixed", "drift", FINAL),
    ("H5 part", "Sudden penalty (Fixed - Sudden, 255k-350k)", "sac", "fixed", "sudden", SUDDEN),
]
hyp = ["| Test | Algorithm | Comparison | Seeds | Mean difference [95% CI] | MDE | Verdict |",
       "|---|---|---|---|---|---|---|"]
diffs = {}
for hid, lab, algo, ca, cb, win in CONTRASTS:
    ss, d = paired(algo, ca, cb, win)
    if not len(d):
        continue
    ci = boot_ci(d)
    diffs[(algo, lab)] = (ss, d, ci, mde(d))
    md_ = f"{mde(d):.0f}" if len(d) >= 3 else "-"
    v = "check (not a hypothesis)" if hid == "Check" else verdict(ci, len(d))
    hyp.append(f"| {hid} | {dict(ALGOS)[algo]} | {lab} | {len(d)} | {fmt(d.mean(), ci, 1)} | {md_} | {v} |")
for hid, lab, ca, cb, win in [("H2", "Drift penalty, CrossQ minus SAC", "fixed", "drift", FINAL),
                              ("H5", "Sudden penalty, CrossQ minus SAC", "fixed", "sudden", SUDDEN)]:
    s1, d1 = paired("crossq", ca, cb, win)
    s2, d2 = paired("sac", ca, cb, win)
    common = sorted(set(s1) & set(s2))
    if not common:
        continue
    did = np.array([d1[s1.index(s)] - d2[s2.index(s)] for s in common])
    ci = boot_ci(did)
    md_ = f"{mde(did):.0f}" if len(did) >= 3 else "-"
    hyp.append(f"| {hid} | both | {lab} | {len(did)} | {fmt(did.mean(), ci, 1)} | {md_} | {verdict(ci, len(did))} |")
hyp.append("\nPositive = the second condition scores lower (a penalty). Verdict rule (pre-registered): "
           "supported if the 95% CI is entirely above 0, contradicted if entirely below 0. MDE = minimum "
           "detectable effect (80% power, paired t-test, alpha 0.05).")
open("tables/hypotheses.md", "w").write("\n".join(hyp) + "\n")


# ---------------------------------------------------------------- Table 3: settings and compute
def meta(path):
    out = {}
    for line in open(os.path.join(path, "meta.txt")):
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def pkgs(path):
    f = os.path.join(path, "packages.txt")
    out = {}
    if os.path.exists(f):
        for line in open(f):
            if "==" in line:
                k, v = line.strip().split("==", 1)
                out[k.lower()] = v
    return out


groups = {}
for run_dir in sorted(glob.glob(os.path.join(ROOT, "raw", "*_seed*_*"))):
    name = os.path.basename(run_dir)
    group = re.sub(r"_seed\d+_.*", "", name)
    groups.setdefault(group, []).append(run_dir)
comp = ["| Run group | Runs | Exit 0 | Minutes per run (min-max) | GPU-hours | GPU | Key packages |",
        "|---|---|---|---|---|---|---|"]
total = 0.0
for g, dirs in groups.items():
    ms = [meta(d) for d in dirs]
    mins = [float(m.get("wall_minutes", "nan")) for m in ms]
    ok = sum(m.get("exit_code") == "0" for m in ms)
    p = pkgs(dirs[0])
    keys = ", ".join(f"{k} {p[k]}" for k in ("jax", "mujoco", "gymnasium", "myosuite") if k in p)
    gh = np.nansum(mins) / 60
    total += gh
    comp.append(f"| {g} | {len(dirs)} | {ok} | {np.nanmin(mins):.0f}-{np.nanmax(mins):.0f} | {gh:.1f} | "
                f"{ms[0].get('gpu', '').split(',')[0]} | {keys} |")
comp.append(f"\nTotal: {total:.1f} GPU-hours over {sum(len(v) for v in groups.values())} runs "
            "(one run per GPU; Kaggle sessions with 2 GPUs run two at a time). Python "
            f"{meta(next(iter(groups.values()))[0]).get('python', '').replace('Python ', '')}. "
            "Hyperparameters: see `configs/` (TRAIN_ARGS); summary below.")
cq = open("configs/main/handreach_crossq_fixed.cfg").read()
sac = open("configs/main/handreach_sac_fixed.cfg").read()


def args(cfg):
    a = re.search(r'TRAIN_ARGS="([^"]+)"', cfg).group(1).split()
    return {a[i].lstrip("-"): a[i + 1] for i in range(0, len(a) - 1, 2)}


A, B = args(cq), args(sac)
comp += ["", "| Setting | CrossQ | SAC |", "|---|---|---|"]
for k in ["n_neurons", "crossq_style", "bn", "bn_mode", "bn_momentum", "policy_delay", "adam_b1", "lr", "utd",
          "n_critics", "critic_activation"]:
    b = B.get(k)
    if k in ("bn_mode", "bn_momentum") and B.get("bn") == "0":
        b = "n/a (no BN)"
    comp.append(f"| {k} | {A.get(k)} | {b} |")
comp.append("| target network | none (tau = 1, joint forward pass) | tau = 0.005 |")
comp.append("| shared | 500k steps; learning starts 5,000; buffer 1M; batch 256 (sbx/sac/sac.py default); "
            "eval every 5,000 steps, 10 episodes | same |")
open("tables/settings_compute.md", "w").write("\n".join(comp) + "\n")


# ---------------------------------------------------------------- Figure: schedules
fig, ax = plt.subplots(figsize=(3.4, 1.7))
t = np.linspace(0, 500_000, 1001)
sched = {"nofatigue": 0 * t, "fixed": 0.5 + 0 * t, "drift": 0.5 * np.clip(t / 400_000, 0, 1),
         "sudden": np.where(t > 250_000, 0.5, 0.0)}
offset = {"nofatigue": 0, "fixed": 0.006, "drift": 0, "sudden": -0.006}  # keep overlapping lines visible
for c, _, col in CONDS:
    ax.plot(t / 1000, sched[c] + offset[c], color=col, linewidth=1.8, label=LABEL[c],
            linestyle=(0, (4, 2)) if c == "sudden" else "-")
ax.set_xlabel("Training steps (thousands)")
ax.set_ylabel("Fatigued fraction")
ax.set_yticks([0, 0.25, 0.5])
ax.set_ylim(-0.04, 0.6)
ax.grid(axis="y", color=GRID, linewidth=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=2, frameon=False)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/schedules.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- Figure: learning curves
def band(algo, cond, smooth=3):
    ss = seeds(runs, algo, cond)
    if not ss:
        return None
    L = min(len(runs[(algo, cond, s)]["r"]) for s in ss)
    Y = np.array([np.convolve(runs[(algo, cond, s)]["r"][:L], np.ones(smooth) / smooth, "valid") for s in ss])
    tt = runs[(algo, cond, ss[0])]["t"][:L][smooth // 2: smooth // 2 + Y.shape[1]]
    if len(ss) >= 3:
        B = np.random.default_rng(0).integers(0, len(ss), (2000, len(ss)))
        means = Y[B].mean(1)
        lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    else:
        lo = hi = None
    return tt, Y.mean(0), lo, hi, len(ss)


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for ax, (a, al) in zip(axes, ALGOS):
    ns = []
    for c, cl, col in CONDS:
        b = band(a, c)
        if b is None:
            continue
        tt, m, lo, hi, n = b
        ns.append(f"{cl}: {n}")
        if lo is not None:
            ax.fill_between(tt / 1000, lo, hi, color=col, alpha=0.18, linewidth=0)
        ax.plot(tt / 1000, m, color=col, linewidth=1.6, label=cl)
    if "0" in random_ref:
        ax.axhline(random_ref["0"], color=MUTED, linestyle=(0, (3, 3)), linewidth=1)
        ax.text(495, random_ref["0"] + 10, "random policy", color=MUTED, fontsize=7, ha="right")
    counts = sorted({x.split(": ")[1] for x in ns})
    ax.set_title(f"{al} ({'/'.join(counts)} seeds per condition)", fontsize=9)
    ax.set_xlabel("Training steps (thousands)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("Evaluation return")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=4, frameon=False)
fig.subplots_adjust(wspace=0.08)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/main_curves.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)


# ---------------------------------------------------------------- Figure: paired differences
items = [(k, v) for k, v in diffs.items()]
fig, ax = plt.subplots(figsize=(3.5, 0.45 * len(items) + 0.8))
ylabels = []
for i, ((algo, lab), (ss, d, ci, md_)) in enumerate(items):
    y = len(items) - 1 - i
    ylabels.append((y, f"{dict(ALGOS)[algo]}: {lab.split(' (')[0]}\n(n = {len(d)})"))
    if not np.isnan(md_):
        ax.plot([-md_, md_], [y, y], color=GRID, linewidth=7, solid_capstyle="butt", zorder=0)
    jitter = np.linspace(-0.12, 0.12, len(d))
    ax.scatter(d, y + jitter, s=12, color=MUTED, zorder=2, linewidths=0)
    if not np.isnan(ci[0]):
        ax.plot(ci, [y, y], color=INK, linewidth=1.6, zorder=3)
    ax.scatter([d.mean()], [y], s=34, color=INK, zorder=4, marker="D")
ax.axvline(0, color=MUTED, linewidth=0.8)
ax.set_yticks([y for y, _ in ylabels])
ax.set_yticklabels([s for _, s in ylabels], fontsize=7)
ax.set_xlabel("Paired difference in return (positive = penalty)")
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_ylim(-0.6, len(items) - 0.4)
# Caption for the paper: dots = seeds; diamond and line = mean and 95% bootstrap CI;
# grey band = +/- minimum detectable effect.
for ext in ("png", "pdf"):
    fig.savefig(f"figures/main_differences.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig)

print(open("tables/main_results.md").read())
print(open("tables/hypotheses.md").read())
print("saved figures/schedules.*, main_curves.*, main_differences.*, tables/main_results.*, "
      "tables/hypotheses.md, tables/settings_compute.md")
