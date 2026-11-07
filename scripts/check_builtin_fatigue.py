"""Measure how much MyoSuite's BUILT-IN fatigue changes a task. CPU only, under a minute.

    python scripts/check_builtin_fatigue.py [out.csv]

Motivation for src/fatigue_drift.py: in MyoSuite's myoFati* tasks, fatigue is reset at every
episode. This script reports, for each task, the episode length and the largest fatigued
fraction (MF) any muscle reaches in one episode when ALL muscles are commanded fully on (the worst
case), and how long continuous maximal effort takes to fatigue a muscle by 20% and by 50%.
"""
import csv
import sys

import gymnasium as gym
import numpy as np
import myosuite  # noqa: F401

TASKS = ["myoFatiElbowPose1D6MRandom-v0", "myoFatiHandReachRandom-v0"]
rows = []
for env_id in TASKS:
    env = gym.make(env_id, seed=0)
    u = env.unwrapped
    env.reset(seed=0)
    steps, mf_max = 0, 0.0
    while True:
        _, _, te, tr, _ = env.step(np.ones(env.action_space.shape))
        steps += 1
        mf_max = max(mf_max, float(u.muscle_fatigue._MF.max()))
        if te or tr:
            break
    episode_s = steps * u.dt
    env.reset(seed=1)
    mf_after_reset = float(u.muscle_fatigue._MF.max())

    f = u.muscle_fatigue
    f.reset()
    t, marks = 0.0, {}
    while t < 3600 and len(marks) < 2:
        f.compute_act(np.ones(f.na))
        t += f._dt
        for lvl in (0.2, 0.5):
            if f._MF.max() >= lvl and lvl not in marks:
                marks[lvl] = t
    rows.append({
        "task": env_id,
        "max_episode_seconds": round(env.spec.max_episode_steps * u.dt, 2),
        "episode_steps_at_max_effort": steps,  # can end early: terminates if the task fails
        "episode_seconds_at_max_effort": round(episode_s, 2),
        "max_fatigued_fraction_in_episode": round(mf_max, 4),
        "fatigued_fraction_after_reset": round(mf_after_reset, 4),
        "seconds_max_effort_to_20pct": round(marks.get(0.2, float("nan")), 1),
        "seconds_max_effort_to_50pct": round(marks.get(0.5, float("nan")), 1),
    })

for r in rows:
    print(r)
if len(sys.argv) > 1:
    with open(sys.argv[1], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("saved", sys.argv[1])
