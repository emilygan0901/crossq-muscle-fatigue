"""Evaluate the final policy of a run at a range of fixed fatigue levels (Amendment 4, exploratory).

Called from CrossQ's train.py after training when FATIGUE_SWEEP is set (third coursework edit in
patches/crossq_coursework.patch). For each level in SWEEP_LEVELS, the final policy is run
deterministically for SWEEP_EPISODES episodes with every muscle's fatigued fraction fixed at that
level. The evaluation environment is created with the same seed at every level, so every level
sees the same sequence of targets. Results go to fatigue_sweep.npz in the run folder:

    levels   (L,)      fatigue levels tested
    returns  (L, N)    episode returns
    solved   (L, N)    1 if the episode ended solved, else 0
    seed     ()        evaluation-environment seed

If SAVE_MODEL is set, the actor's weights and BRN statistics are also saved (final_actor.pkl).
Settings (environment variables): SWEEP_LEVELS (default 0,0.2,0.4,0.5,0.6,0.8), SWEEP_EPISODES (default 50).
"""
import os
import sys

import numpy as np
import gymnasium as gym

DEFAULT_LEVELS = "0,0.2,0.4,0.5,0.6,0.8"
SEED_OFFSET = 100_000   # keeps sweep targets separate from the training and evaluation envs


def run(model, env_id, seed, out_dir="."):
    levels = [float(x) for x in os.environ.get("SWEEP_LEVELS", DEFAULT_LEVELS).split(",")]
    n_ep = int(os.environ.get("SWEEP_EPISODES", "50"))
    eval_seed = SEED_OFFSET + int(seed)
    before = os.environ.get("FATIGUE_FIXED")
    returns = np.zeros((len(levels), n_ep))
    solved = np.zeros((len(levels), n_ep))
    try:
        for i, lvl in enumerate(levels):
            os.environ["FATIGUE_FIXED"] = str(lvl)          # read by fatigue_drift at every reset
            env = gym.make(env_id, seed=eval_seed)            # same targets at every level
            for j in range(n_ep):
                obs, info = env.reset()
                total, done = 0.0, False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, r, term, trunc, info = env.step(action)
                    total += float(r)
                    done = term or trunc
                returns[i, j] = total
                solved[i, j] = float(bool(info.get("is_success", False)))
            env.close()
            print(f"[fatigue_sweep] level {lvl:.2f}: return {returns[i].mean():.1f}, "
                  f"solved {solved[i].mean():.2f} ({n_ep} episodes)", file=sys.stderr)
    finally:
        if before is None:
            os.environ.pop("FATIGUE_FIXED", None)
        else:
            os.environ["FATIGUE_FIXED"] = before
    np.savez(os.path.join(out_dir, "fatigue_sweep.npz"), levels=np.array(levels), returns=returns,
             solved=solved, seed=eval_seed)
    if os.environ.get("SAVE_MODEL"):
        save_actor(model, out_dir)


def save_actor(model, out_dir):
    """Save only the actor's weights and BRN statistics (a few MB), not the full model with its
    2048-unit critics and optimiser states (about 400 MB)."""
    try:
        import pickle
        import jax
        state = model.policy.actor_state
        payload = {"params": jax.device_get(state.params),
                   "batch_stats": jax.device_get(getattr(state, "batch_stats", None))}
        with open(os.path.join(out_dir, "final_actor.pkl"), "wb") as fh:
            pickle.dump(payload, fh)
    except Exception as exc:  # saving is a convenience; never lose the sweep because of it
        print(f"[fatigue_sweep] actor not saved: {exc}", file=sys.stderr)
