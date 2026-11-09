# How to run the experiments

Everything is driven by **config files** (`configs/`) and scripts (`scripts/`).
A run is never started by typing training flags by hand.

## Layout

```
configs/repro/          reproduction of the CrossQ paper (Walker2d, Fig. 5): authors' code, unmodified
configs/pilot/          Hand Reach pilot: fatigue off vs fatigue drift, SAC vs CrossQ (1 seed)
configs/main/           main experiment: SAC/CrossQ x no fatigue / fixed 0.5 / drift 0->0.5 (seeds 11-15)
configs/bridge/         Walker2d reproduction settings run in the MyoSuite software setup
HYPOTHESIS.md           pre-registered hypotheses, metrics, decision rules and batch plan
tests/test_fatigue_drift.py       checks of src/fatigue_drift.py (CPU, ~1 min)
scripts/check_builtin_fatigue.py  measures how little MyoSuite's built-in fatigue changes a task
src/fatigue_drift.py    OUR CODE: fatigue that increases over training (registers *Drift-v0 tasks)
patches/crossq_coursework.patch   OUR CHANGES to the authors' train.py (2 small edits)
scripts/run_experiment.sh   runs ONE config with ONE seed and records everything
scripts/run_batch.sh        runs several config:seed jobs over the available GPUs
scripts/random_baseline.sh  random-policy reference for a config's task
scripts/analyze_repro.py    reproduction figure and table from the raw results
scripts/analyze_h1.py       primary test H1 (Fixed vs Drift, final 100k steps)
scripts/analyze_ramp.py     secondary metric: largest drop in return during the ramp
scripts/analyze_main.py     main figures and tables (curves, paired differences, results, hypotheses, compute)
results/raw/<run_id>/   one folder per run (created automatically)
results/run_log.md      one line per run (created automatically)
notes/run_notes.md      what was run, why, and what happened (written by hand)
```

## Code used

| Folder | What | Used for |
|---|---|---|
| `CrossQ-baseline/` | authors' code (github.com/adityab/CrossQ), commit `a318a266679bec3f9d93c44e029860a6da077c89`, **unmodified** | reproduction (`configs/repro/`) |
| `CrossQ-dev/` | same commit **+ `patches/crossq_coursework.patch`** | MyoSuite experiments (`configs/pilot/`, later `configs/main/`) |

The patch makes exactly two changes to `train.py`:
1. `import fatigue_drift`, which registers MyoSuite and our fatigue-drift tasks (`src/` is on `PYTHONPATH`).
2. The number of evaluation episodes is read from `N_EVAL_EPISODES` (default 1, as in the original).
   MyoSuite tasks draw random targets, so we evaluate on 10 episodes.

Each run records the code commit, the number of changed files (`code_files_changed`) and the
full diff (`code.diff`), so it is always visible which code a result came from.

## Fatigue drift (our contribution's experimental condition)

MyoSuite's built-in fatigue (`myoFati*` tasks) resets at every 2-second episode and builds up far
too slowly to matter (about 25 s of maximal effort for 20% fatigue). `src/fatigue_drift.py`
instead starts each episode with a prescribed fatigued fraction of every muscle, which follows a
schedule over **training** steps. See the docstring in that file for details. Settings come from
the config's `ENV_VARS`:

| Variable | Meaning |
|---|---|
| `FATIGUE_MAX` | final fatigued fraction (0 = control: fatigue off) |
| `FATIGUE_RAMP_START`, `FATIGUE_RAMP_STEPS` | linear ramp from 0 to `FATIGUE_MAX` |
| `FATIGUE_FIXED` | constant level instead of the schedule (the Fixed condition; random baseline) |
| `N_EVAL_EPISODES` | evaluation episodes per checkpoint |

The control (`FATIGUE_MAX=0`) uses the same task, so both conditions share MyoSuite's fatigue
activation dynamics and differ only in the fatigue level. Each environment is constructed with a
seed derived from the run's seed (`RUN_SEED`), so MyoSuite's random targets are repeatable.

## Software environments

**Reproduction (`env-repro`)**, installed in the order given in the authors' README:
1. Python 3.11.5 virtual environment
2. `pip install -e CrossQ-baseline` (pins mujoco 2.3.7, gymnasium 0.29.1, torch 2.1.2, ...)
3. `pip install "jax[cuda12_pip]==0.4.19" "nvidia-cudnn-cu12==8.9.7.29" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

Step 3 upgrades shared NVIDIA libraries beyond torch 2.1.2's pins (pip prints a warning). torch is
only imported by Stable-Baselines3, never used for computation. The cuDNN pin is our addition.

**MyoSuite experiments (`env-dev`)**: Python 3.11.5; `CrossQ-dev` + `myosuite==2.8.4` +
`jax[cuda12_pip]==0.4.19`, with mujoco overridden to 3.1.2 (CrossQ pins 2.3.7, which no
gymnasium-based MyoSuite release supports). MyoSuite pulls in torch 2.0.1, whose CUDA 11
libraries overwrite files JAX needs, so `nvidia-cusolver-cu12` and
`nvidia-cudnn-cu12==8.9.7.29` are reinstalled last.

## Commands

```bash
# one run
scripts/run_experiment.sh configs/repro/walker2d_crossq.cfg 1 0     # config, seed, GPU index

# several runs, spread over the GPUs (PER_GPU runs share each GPU)
PER_GPU=1 scripts/run_batch.sh configs/repro/walker2d_sac.cfg:1 configs/repro/walker2d_crossq.cfg:1

# random-policy reference, 100 episodes (drift tasks need a fixed fatigue level)
scripts/random_baseline.sh configs/repro/walker2d_sac.cfg 100 0
FATIGUE_FIXED=0.6 scripts/random_baseline.sh configs/pilot/handreach_sac_drift.cfg 100 0

# checks of the fatigue-drift module, and the built-in fatigue measurement (MyoSuite environment)
PYTHONPATH=src python tests/test_fatigue_drift.py
python scripts/check_builtin_fatigue.py results/builtin_fatigue.csv

# reproduction figure and table from the raw results (needs numpy + matplotlib)
python scripts/analyze_repro.py results     # -> figures/repro_walker2d.*, tables/repro_walker2d.*
python scripts/analyze_h1.py results        # -> tables/h1_crossq.md (primary test H1)
python scripts/analyze_ramp.py results      # -> figures/ramp_crossq.*, tables/ramp_crossq.md (dip during the ramp)
python scripts/analyze_main.py results      # -> figures/{schedules,main_curves,main_differences}.*, tables/{main_results,hypotheses,settings_compute}.*
```

## What each run folder contains

| File | Content |
|---|---|
| `config.cfg` | copy of the config used |
| `meta.txt` | run id, env vars, exact command, code commit, changed files, Python, GPU, start/end, minutes, exit code |
| `code.diff` | changes to the code directory (empty for the reproduction; the patch for MyoSuite runs) |
| `packages.txt` | exact versions of every installed package |
| `train.log` | everything the training script printed (including the fatigue schedule used) |
| `evaluations.npz` | evaluation returns (and, for MyoSuite, success = solved) over training: the raw result |
| `eval_logs/`, `logs/` | the training script's own output (evaluation logs, TensorBoard) |

Note: `train.py` names its output folders `SAC_...` for both algorithms, because the authors
launch CrossQ as `-algo sac` with CrossQ settings. The run id and `config.cfg` say which is which.
