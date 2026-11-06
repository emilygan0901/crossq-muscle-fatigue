# How to run the experiments

Everything is driven by **config files** (`configs/`) and two scripts (`scripts/`).
A run is never started by typing training flags by hand.

## Layout

```
configs/repro/         reproduction of the CrossQ paper (Walker2d, Fig. 5)
scripts/run_experiment.sh   runs ONE config with ONE seed and records everything
scripts/run_batch.sh        runs several config:seed jobs over the available GPUs
results/raw/<run_id>/  one folder per run (created automatically)
results/run_log.md     one line per run (created automatically); fill in the notes column
```

## Code used

- `CrossQ-baseline/`: the authors' code (github.com/adityab/CrossQ), commit
  `a318a266679bec3f9d93c44e029860a6da077c89`, **unmodified**. Used for the reproduction.
- Each run records the commit and the number of changed files (`code_files_changed`) in
  its `meta.txt`, so it can be shown that the reproduction used unmodified code.

## Software environment for the reproduction

Installed in the order given in the authors' README:

1. Python 3.11.5 virtual environment
2. `pip install -e CrossQ-baseline` (pins mujoco 2.3.7, gymnasium 0.29.1, torch 2.1.2, ...)
3. `pip install "jax[cuda12_pip]==0.4.19" "nvidia-cudnn-cu12==8.9.7.29" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

Step 3 upgrades several shared NVIDIA libraries beyond torch 2.1.2's exact pins (pip prints a
dependency warning about torch). This is expected: torch is only imported by
Stable-Baselines3 and never used for computation; JAX does the training.
The cuDNN pin (8.9.7.29) is an addition to the README.

## Commands

```bash
# one run
scripts/run_experiment.sh configs/repro/walker2d_crossq.cfg 1 0     # config, seed, GPU index

# several runs, spread over the GPUs (PER_GPU runs share each GPU)
PER_GPU=1 scripts/run_batch.sh configs/repro/walker2d_sac.cfg:1 configs/repro/walker2d_crossq.cfg:1

# random-policy reference ("learns above random"), 100 episodes, for a config's environment
scripts/random_baseline.sh configs/repro/walker2d_sac.cfg 100 0     # -> results/random/
```

## What each run folder contains

| File | Content |
|---|---|
| `config.cfg` | copy of the config used |
| `meta.txt` | run id, exact command, code commit, changed files, Python, GPU, start/end, minutes, exit code |
| `code.diff` | changes to the code directory (empty for the reproduction) |
| `packages.txt` | exact versions of every installed package |
| `train.log` | everything the training script printed |
| `evaluations.npz` | evaluation returns over training: the raw result |
| `eval_logs/`, `logs/` | the training script's own output (evaluation logs, TensorBoard) |

Note: `train.py` names its output folders `SAC_...` for both algorithms, because the authors
launch CrossQ as `-algo sac` with CrossQ settings. The run id and `config.cfg` say which is which.
