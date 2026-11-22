"""Observed muscle activation under a full command, with and without fatigue (paper Section III-A).

    python scripts/check_activation_saturation.py   # -> results/activation_saturation.csv

Shows that fatigue is partly observable: the state includes each muscle's current activation, which
saturates at 1 - MF. Each fatigue level runs in a fresh process because the wrapper reads its
settings from environment variables. Needs the MyoSuite environment (env-dev). CPU only, seconds.
"""
import os
import subprocess
import sys

PROBE = r"""
import os, sys, numpy as np
sys.path.insert(0, "src")
import fatigue_drift  # noqa: F401  (registers the drift task)
from myosuite.utils import gym
env = gym.make("myoFatiHandReachRandomDrift-v0")
obs, _ = env.reset(seed=0)
a = np.ones(env.action_space.shape)
for _ in range(30):
    obs, *_ = env.step(a)
act = obs[-39:]   # the last 39 observation entries are the muscle activations
print(f"{os.environ['FATIGUE_FIXED']},{act.mean():.3f},{act.max():.3f}")
"""
rows = ["fatigue_level,mean_activation_after_30_steps_of_full_command,max_activation"]
for lvl in ("0", "0.5"):
    out = subprocess.run([sys.executable, "-c", PROBE], env={**os.environ, "FATIGUE_FIXED": lvl},
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    rows.append(out)
os.makedirs("results", exist_ok=True)
open("results/activation_saturation.csv", "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
