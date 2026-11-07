"""Measure the return of a uniformly random policy: the 'learns above random' reference.

    python random_baseline.py <env_id> <n_episodes> <seed> <out.npz>

Actions are sampled uniformly from the environment's action space. Episodes end the same way
as in training (termination or the environment's time limit).
"""
import sys

import gymnasium as gym
import numpy as np

env_id, n_episodes, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
if "Drift" in env_id:
    import fatigue_drift  # noqa: F401  (registers the fatigue-drift tasks; needs src/ on PYTHONPATH)
elif env_id.startswith("myo"):
    import myosuite  # noqa: F401  (registers MyoSuite environments)

env = gym.make(env_id)
env.action_space.seed(seed)
returns, lengths = [], []
for ep in range(n_episodes):
    env.reset(seed=seed + ep)
    total, steps, done = 0.0, 0, False
    while not done:
        _, r, terminated, truncated, _ = env.step(env.action_space.sample())
        total += float(r)
        steps += 1
        done = terminated or truncated
    returns.append(total)
    lengths.append(steps)

returns, lengths = np.array(returns), np.array(lengths)
np.savez(out, returns=returns, lengths=lengths, env_id=env_id, seed=seed)
print(f"{env_id}: random policy over {n_episodes} episodes: "
      f"return mean {returns.mean():.1f} (sd {returns.std():.1f}), "
      f"median {np.median(returns):.1f}, mean length {lengths.mean():.0f}")
