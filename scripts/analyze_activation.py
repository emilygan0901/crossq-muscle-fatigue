"""How the saved Amendment 4 policies use their muscles at each fatigue level (exploratory).

    CODE_DIR=/path/to/CrossQ-dev python scripts/analyze_activation.py [results_root]
        # -> results/activation_rollouts.npz, tables/activation_crossq.md, figures/policy_sweep.*

Rebuilds CrossQ's actor with the training settings, loads each run's saved actor
(final_actor.pkl, written by src/fatigue_sweep.py) and replays the final-policy sweep on the CPU:
the same evaluation seed, levels and number of episodes as fatigue_sweep.npz, so the returns can be
checked against the GPU sweep. For every step it records the commanded drive of each muscle
(MyoSuite maps the action a in [-1, 1] to sigmoid(5(a - 0.5))), the muscle's actual activation, and
the distance to the targets. Per episode it reports:

  drive       mean commanded drive over muscles and steps (what the policy asks for)
  activation  mean actual activation over muscles and steps (what the muscles deliver)
  at_cap      mean fraction of muscles within 5% of their available capacity 1 - MF
  final_dist  distance to the targets at the last step (all five fingertips, metres)
  t_solve     first step at which the reach is solved (episodes that are solved only)

Needs the MyoSuite environment and CPU only (JAX_PLATFORMS=cpu is set here).
"""
import glob
import os
import pickle
import re
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = sys.argv[1] if len(sys.argv) > 1 else "results"
CODE_DIR = os.environ.get("CODE_DIR", "/kaggle/working/CrossQ-dev")
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, "src")

import numpy as np  # noqa: E402
import jax  # noqa: E402
import optax  # noqa: E402
import flax.linen as nn  # noqa: E402
import gymnasium as gym  # noqa: E402
import fatigue_drift  # noqa: E402,F401  (registers myoFatiHandReachRandomDrift-v0)
from sbx import SAC  # noqa: E402
from sbx.sac.utils import activation_fn as ACT  # noqa: E402  (same mapping as train.py)

ENV_ID = "myoFatiHandReachRandomDrift-v0"
N_EP = int(os.environ.get("ACT_EPISODES", "50"))


def build_model(env):
    """CrossQ with the exact training settings of configs/sweep/*.cfg (see train.py)."""
    return SAC("MlpPolicy", env,
               policy_kwargs=dict(activation_fn=ACT["relu"], layer_norm=False, batch_norm=True,
                                  batch_norm_momentum=0.99, batch_norm_mode="brn_actor", dropout_rate=None,
                                  n_critics=2, net_arch={"pi": [256, 256], "qf": [2048, 2048]},
                                  optimizer_class=optax.adam, optimizer_kwargs=dict(b1=0.5, b2=0.999)),
               gradient_steps=1, policy_delay=3, crossq_style=True, td3_mode=False,
               use_bnstats_from_live_net=False, policy_q_reduce_fn=jax.numpy.min, learning_starts=5000,
               learning_rate=1e-3, qf_learning_rate=1e-3, tau=1.0, gamma=0.99, verbose=0,
               buffer_size=1000, seed=0)


def load_actor(model, path):
    p = pickle.load(open(path, "rb"))
    st = model.policy.actor_state
    model.policy.actor_state = st.replace(params=p["params"], batch_stats=p["batch_stats"])


def rollout(model, level, seed):
    os.environ["FATIGUE_FIXED"] = str(level)
    env = gym.make(ENV_ID, seed=seed)                 # same targets as the GPU sweep
    u = env.unwrapped
    near = len(u.tip_sids) * 0.0125
    out = {k: [] for k in ("ret", "solved", "drive", "activation", "at_cap", "final_dist", "t_solve")}
    for _ in range(N_EP):
        obs, info = env.reset()
        cap = 1.0 - np.asarray(u.muscle_fatigue.MF)
        R, done, t, ts = 0.0, False, 0, np.nan
        drive, act, atcap = [], [], []
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            drive.append(1.0 / (1.0 + np.exp(-5.0 * (np.asarray(a) - 0.5))))
            obs, r, term, trunc, info = env.step(a)
            R += float(r)
            done = term or trunc
            m = np.asarray(u.sim.data.act).copy()
            act.append(m)
            atcap.append(np.mean(m >= 0.95 * cap))
            dist = float(np.linalg.norm(u.obs_dict["reach_err"]))
            t += 1
            if np.isnan(ts) and dist < near:
                ts = t
        out["ret"].append(R)
        out["solved"].append(float(bool(info.get("is_success", False))))
        out["drive"].append(np.mean(drive))
        out["activation"].append(np.mean(act))
        out["at_cap"].append(np.mean(atcap))
        out["final_dist"].append(dist)
        out["t_solve"].append(ts)
    env.close()
    return {k: np.array(v) for k, v in out.items()}


runs = sorted(glob.glob(os.path.join(ROOT, "raw", "sweep_handreach_crossq_*_seed*_*", "final_actor.pkl")))
if not runs:
    sys.exit("no final_actor.pkl found")
results, check = {}, []
SAVED = os.path.join(ROOT, "activation_rollouts.npz")
if os.environ.get("REUSE") and os.path.exists(SAVED):      # redraw from a previous replay
    z = np.load(SAVED)
    for i, k in enumerate(z["keys"]):
        c, s_, l = str(k).split("|")
        results[(c, int(s_), float(l))] = {m: z[m][i] for m in z.files if m != "keys"}
    check = list(z["check"]) if "check" in z.files else [np.nan]
    runs_to_replay = []
else:
    os.environ["FATIGUE_FIXED"] = "0"
    model = build_model(gym.make(ENV_ID))
    runs_to_replay = runs
for path in runs_to_replay:
    run = os.path.dirname(path)
    cond, seed = re.match(r".*sweep_handreach_crossq_(\w+?)_seed(\d+)_", run).groups()
    sw = np.load(os.path.join(run, "fatigue_sweep.npz"))
    load_actor(model, path)
    for i, lvl in enumerate(sw["levels"]):
        res = rollout(model, float(lvl), int(sw["seed"]))
        results[(cond, int(seed), float(lvl))] = res
        check.append(abs(res["ret"].mean() - sw["returns"][i].mean()))
        print(f"{cond} seed {seed} level {lvl:.1f}: return {res['ret'].mean():.1f} (GPU sweep "
              f"{sw['returns'][i].mean():.1f}), drive {res['drive'].mean():.3f}, activation "
              f"{res['activation'].mean():.3f}, at cap {res['at_cap'].mean():.3f}", file=sys.stderr)

keys = sorted(results)
if runs_to_replay:
  np.savez(SAVED, keys=np.array([f"{c}|{s}|{l}" for c, s, l in keys]), check=np.array(check),
           **{f"{m}": np.array([results[k][m] for k in keys]) for m in results[keys[0]]})

# ---------------------------------------------------------------- table
levels = sorted({k[2] for k in keys})
conds = [("nofatigue", "Trained without fatigue"), ("fixed", "Trained at fatigue 0.5")]
md = ["Mean over seeds (11-13) of each policy's mean over the episodes at each test level (CPU replay of the final-policy sweep).",
      "", "| Training condition | Measure | " + " | ".join(f"{l:g}" for l in levels) + " |",
      "|---|---|" + "---|" * len(levels)]
labels = [("drive", "commanded drive", "{:.3f}"), ("activation", "actual activation", "{:.3f}"),
          ("at_cap", "fraction of muscles at capacity", "{:.3f}"), ("final_dist", "final distance (m)", "{:.3f}"),
          ("t_solve", "steps to solve (solved episodes)", "{:.0f}"), ("solved", "solved fraction", "{:.2f}")]
summary = {}
for c, lab in conds:
    ss = sorted({k[1] for k in keys if k[0] == c})
    for m, mlab, f in labels:
        vals = [np.mean([np.nanmean(results[(c, s, l)][m]) for s in ss]) for l in levels]
        summary[(c, m)] = (ss, vals)
        md.append(f"| {lab} | {mlab} | " + " | ".join(f.format(v) for v in vals) + " |")
md.append(f"\nCPU replay check: mean absolute difference from the GPU sweep returns {np.mean(check):.1f} "
          f"(max {np.max(check):.1f}) over {len(check)} policy-level pairs.")
md.append("Exploratory (Amendment 4 policies, 3 seeds per training condition).")
os.makedirs("tables", exist_ok=True)
open("tables/activation_crossq.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

# ---------------------------------------------------------------- figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "pdf.fonttype": 42,
                     "axes.edgecolor": "#6b6a66", "xtick.color": "#6b6a66", "ytick.color": "#6b6a66"})
COL = {"nofatigue": "#1baf7a", "fixed": "#2a78d6"}
# return from the GPU sweep (fatigue_sweep.npz), the rest from the CPU replay
gpu_ret = {}
for path in runs:
    run = os.path.dirname(path)
    c, s_ = re.match(r".*sweep_handreach_crossq_(\w+?)_seed(\d+)_", run).groups()
    gpu_ret[(c, int(s_))] = np.load(os.path.join(run, "fatigue_sweep.npz"))["returns"].mean(1)
fig, axes = plt.subplots(2, 2, figsize=(3.45, 3.3), sharex=True)
panels = (("ret", "Return"), ("t_solve", "Steps to solve"), ("drive", "Commanded drive"),
          ("activation", "Actual activation"))
for ax, (m, ylab), tag in zip(axes.flat, panels, "abcd"):
    for c, lab in conds:
        ss = sorted({k[1] for k in keys if k[0] == c})
        per = [gpu_ret[(c, s)] if m == "ret" else [np.nanmean(results[(c, s, l)][m]) for l in levels] for s in ss]
        for y in per:
            ax.plot(levels, y, color=COL[c], lw=0.6, alpha=0.35)
        ax.plot(levels, np.mean(per, 0), color=COL[c], lw=1.5, marker="o", ms=2.5, label=lab)
    ax.axvline(0.5, color="#6b6a66", ls=(0, (2, 2)), lw=0.8)
    ax.set_ylabel(ylab)
    ax.set_title(tag, loc="left", fontsize=9, fontweight="bold", pad=2)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for ax in axes[1]:
    ax.set_xlabel("Test fatigue level")
h, lb = axes[0, 0].get_legend_handles_labels()
fig.legend(h, lb, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=1, frameon=False, fontsize=7)
fig.subplots_adjust(wspace=0.5, hspace=0.3)
os.makedirs("figures", exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"figures/policy_sweep.{ext}", dpi=200, bbox_inches="tight")
