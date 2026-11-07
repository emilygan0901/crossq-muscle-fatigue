"""Checks that src/fatigue_drift.py does what the methodology says. CPU only, about 1 minute.

    PYTHONPATH=src python tests/test_fatigue_drift.py

Prints PASS/FAIL per check and exits with status 1 if any check fails.
"""
import os
import sys

os.environ.update(FATIGUE_MAX="0.5", FATIGUE_RAMP_START="0", FATIGUE_RAMP_STEPS="2000", RUN_SEED="")
os.environ.pop("FATIGUE_FIXED", None)

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import fatigue_drift as fd  # noqa: E402

DRIFT = "myoFatiHandReachRandomDrift-v0"
BASE = "myoFatiHandReachRandom-v0"
FAILED = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILED.append(name)


# 1-3: the first env created is the training clock; evaluation envs only read it
train, ev = gym.make(DRIFT), gym.make(DRIFT)
check("first env is the training clock, second is not",
      train.get_wrapper_attr("is_clock") and not ev.get_wrapper_attr("is_clock"))
train.reset(seed=0)
check("fatigue level 0 at the start of training", np.allclose(train.unwrapped.muscle_fatigue._MF, 0))
seen_success = False
for _ in range(1000):
    _, _, te, tr, info = train.step(train.action_space.sample())
    if te or tr:
        seen_success |= "is_success" in info
        train.reset()
check("is_success (= MyoSuite 'solved') reported at episode end", seen_success)
check("clock counted 1000 training steps", fd._CLOCK["steps"] == 1000)

# 4-5: evaluation env follows the clock without advancing it
_, info = ev.reset(seed=1)
check(f"eval env reads the clock: level {info['fatigue_level']:.3f} == 0.25",
      abs(info["fatigue_level"] - 0.25) < 1e-9)
check("eval reset sets fatigued fraction MF = 0.25 in every muscle",
      np.allclose(ev.unwrapped.muscle_fatigue._MF, 0.25))
for _ in range(50):
    ev.step(ev.action_space.sample())
check("evaluation steps do not advance the clock", fd._CLOCK["steps"] == 1000)

# 6: fatigue limits force capacity: active fraction MA <= 1 - MF
ev.reset(seed=2)
mas = []
for _ in range(30):
    _, _, te, tr, _ = ev.step(np.ones(ev.action_space.shape))
    mas.append(ev.unwrapped.muscle_fatigue._MA.max())
    if te or tr:
        break
check(f"max active fraction at 25% fatigue = {max(mas):.3f} (must be <= 0.75)", max(mas) <= 0.75 + 1e-6)

# 7: ramp-then-hold: level saturates at FATIGUE_MAX after the ramp
fd._CLOCK["steps"] = 10_000
_, info = ev.reset(seed=3)
check(f"level held at FATIGUE_MAX after the ramp: {info['fatigue_level']:.2f}", abs(info["fatigue_level"] - 0.5) < 1e-9)

# 8: fixed condition: FATIGUE_FIXED overrides the schedule at any clock value
os.environ["FATIGUE_FIXED"] = "0.5"
fd._CLOCK["steps"] = 0
_, info = ev.reset(seed=4)
check("FATIGUE_FIXED=0.5 gives level 0.5 from the first step", abs(info["fatigue_level"] - 0.5) < 1e-9
      and np.allclose(ev.unwrapped.muscle_fatigue._MF, 0.5))
os.environ.pop("FATIGUE_FIXED")


# 9: control: fatigue off is identical to MyoSuite's own task (same constructor seed)
def rollout(env_id):
    env = gym.make(env_id, seed=123)
    obs, _ = env.reset(seed=7)
    out = [obs.copy()]
    rng = np.random.default_rng(0)
    for _ in range(60):
        obs, r, te, tr, _ = env.step(rng.uniform(-1, 1, env.action_space.shape))
        out.append(np.append(obs, r))
        if te or tr:
            break
    return out


os.environ["FATIGUE_MAX"] = "0"
a, b = rollout(DRIFT), rollout(BASE)
check("fatigue off: observations and rewards identical to myoFatiHandReachRandom-v0",
      len(a) == len(b) and all(np.allclose(x, y) for x, y in zip(a, b)))

# 10: seeding: RUN_SEED makes MyoSuite's random targets repeatable
os.environ["RUN_SEED"] = "11"
fd._N_CREATED["n"] = 0
e1 = gym.make(DRIFT)
o1, _ = e1.reset(seed=11)
fd._N_CREATED["n"] = 0
e2 = gym.make(DRIFT)
o2, _ = e2.reset(seed=11)
check("RUN_SEED: two runs with the same seed start identically", np.allclose(o1, o2))

print("\nALL PASS" if not FAILED else f"\n{len(FAILED)} FAILED")
sys.exit(1 if FAILED else 0)
