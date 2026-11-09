| Run group | Runs | Exit 0 | Minutes per run (min-max) | GPU-hours | GPU | Key packages |
|---|---|---|---|---|---|---|
| main_handreach_crossq_drift | 5 | 5 | 137-139 | 11.5 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| main_handreach_crossq_fixed | 5 | 5 | 138-143 | 11.7 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| main_handreach_crossq_nofatigue | 5 | 5 | 131-137 | 11.2 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| main_handreach_sac_drift | 2 | 2 | 84-87 | 2.9 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| main_handreach_sac_fixed | 2 | 2 | 79-93 | 2.9 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| main_handreach_sac_nofatigue | 2 | 2 | 82-94 | 2.9 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| pilot_handreach_crossq_drift | 1 | 1 | 171-171 | 2.9 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| pilot_handreach_crossq_nofatigue | 1 | 1 | 171-171 | 2.9 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| pilot_handreach_crossqln_nofatigue | 1 | 1 | 141-141 | 2.4 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| pilot_handreach_sac_drift | 1 | 1 | 90-90 | 1.5 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| pilot_handreach_sac_nofatigue | 1 | 1 | 90-90 | 1.5 | Tesla T4 | jax 0.4.19, mujoco 3.1.2, gymnasium 0.29.1, myosuite 2.8.4 |
| repro_walker2d_crossq | 5 | 5 | 110-165 | 12.8 | Tesla T4 | jax 0.4.19, mujoco 2.3.7, gymnasium 0.29.1 |
| repro_walker2d_sac | 5 | 5 | 48-49 | 4.1 | Tesla T4 | jax 0.4.19, mujoco 2.3.7, gymnasium 0.29.1 |

Total: 71.0 GPU-hours over 36 runs (one run per GPU; Kaggle sessions with 2 GPUs run two at a time). Python 3.11.5. Hyperparameters: see `configs/` (TRAIN_ARGS); summary below.

| Setting | CrossQ | SAC |
|---|---|---|
| n_neurons | 2048 | 256 |
| crossq_style | 1 | 0 |
| bn | 1 | 0 |
| bn_mode | brn_actor | n/a (no BN) |
| bn_momentum | 0.99 | n/a (no BN) |
| policy_delay | 3 | 1 |
| adam_b1 | 0.5 | 0.9 |
| lr | 0.001 | 0.001 |
| utd | 1 | 1 |
| n_critics | 2 | 2 |
| critic_activation | relu | relu |
| target network | none (tau = 1, joint forward pass) | tau = 0.005 |
| shared | 500k steps; learning starts 5,000; buffer 1M; batch 256 (sbx/sac/sac.py default); eval every 5,000 steps, 10 episodes | same |
