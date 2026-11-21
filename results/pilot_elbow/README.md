# Elbow-task pilots (6 Nov 2026)

Three single-seed pilot runs on MyoSuite's elbow pose task, run on Kaggle (NVIDIA T4) before the
experiment was pre-registered. They are **not used in any reported analysis**. They are kept here
because they are the reason the elbow task was rejected in favour of Hand Reach (paper, Section IV,
"Design pilots").

| Folder | Algorithm | Steps | Wall time | Mean return 0-25k | Mean return 250-300k | Best | First evaluation at 95% of best |
|---|---|---|---|---|---|---|---|
| `crossq_seed1` | CrossQ | 300k | 84 min | 611 | 767 | 786 | 6,500 steps |
| `sac_seed1_first` | SAC, first attempt (flags later found to be wrong) | 300k | 52 min | 29 | 695 | 769 | 96,500 steps |
| `sac_seed1_rerun` | SAC, rerun with corrected flags | 300k | 44 min | 607 | 770 | 785 | 6,000 steps |

Each `evaluations.npz` holds `timesteps`, `results` (10 evaluation episodes per evaluation) and
`ep_lengths` (100 steps per episode), with an evaluation every 500 steps. `train.log` is the
console output of the run. Values in the table are computed from `evaluations.npz`.

**What they showed.** CrossQ and the corrected SAC both reached 95% of their best return within the
first 6,500 steps and stayed there, so the task has no room to show a difference between conditions
(a ceiling effect). MyoSuite's built-in fatigue was also too weak to matter on this task: maximal
activation fatigues a muscle by at most 1.7% within an episode (`results/builtin_fatigue.csv`).
Both findings led to the Hand Reach task and our own fatigue schedule.

**What did not work.** The first SAC run used incorrect flags, which made SAC learn much more slowly
than it should (return 29 over the first 25k steps, against 607 after the fix). It was rerun with
the corrected flags. Its results are kept for completeness and were not used.

**Not recorded.** These runs predate `scripts/run_experiment.sh`, so they have no `config.cfg`,
`meta.txt` or `packages.txt`, and the exact command lines were not saved with the outputs. The task
ID is `myoFatiElbowPose1D6MRandom-v0` as used in this repository's setup checks; the command lines
are in the Kaggle notebook version that produced them.
