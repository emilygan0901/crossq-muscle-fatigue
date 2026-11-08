# Pre-registered hypothesis and analysis plan

Committed **before** any main-experiment run (`configs/main/`). The commit date of this file is the
record that the hypothesis, metrics and decision rules were fixed in advance. Pilot runs
(`results/raw/pilot_*`, seed 1) informed the design and are not used in any reported analysis.

## Coursework task

Does a gradual loss of muscle strength during training (fatigue drift) hurt CrossQ more than a
constant, equally large loss, and is the effect specific to CrossQ compared with SAC?

**Motivation.** CrossQ (Bhatt et al., ICLR 2024) replaces SAC's target network with Batch
Renormalization, which normalises with statistics of the training data. Batch normalization in RL
is known to suffer when the data distribution shifts. With a replay buffer of 1M transitions (more
than the 500k a run produces), every training batch mixes transitions from all past fatigue levels,
so under drift the batch statistics describe a blend of old and new muscle states.

## Setup (fixed in advance)

| Item | Value |
|---|---|
| Task | `myoFatiHandReachRandomDrift-v0` (MyoSuite Hand Reach, 39 muscles; `src/fatigue_drift.py`) |
| Algorithms | SAC and CrossQ, authors' settings (`configs/main/`), code commit a318a26 + `patches/crossq_coursework.patch` |
| Budget | 500,000 environment steps per run, identical for every run |
| Conditions | **No fatigue** (level 0); **Fixed** (level 0.5 from the first step); **Drift** (0 to 0.5 linearly over steps 0-400k, held at 0.5 for 400k-500k) |
| Fatigue level 0.5 | 50% loss of maximal force capacity; upper end of the 30-50% loss reported for a 60-s maximal contraction of a hand muscle (Bigland-Ritchie et al., 1983) |
| Seeds | 11, 12, 13, 14, 15, the same seeds in every condition (paired design) |
| Evaluation | every 5,000 steps, 10 deterministic episodes, at the current training fatigue level |

## Hypotheses

**H1 (primary).** For CrossQ, mean evaluation return over the final 100k steps (steps 400k-500k,
where Fixed and Drift face the identical task at level 0.5) is lower under Drift than under Fixed.

**H2.** The Drift penalty (Fixed minus Drift, same metric) is larger for CrossQ than for SAC.

**H3 (mechanism).** Specified in an amendment to this file, committed before any mechanism run,
after a single-seed pilot of CrossQ with LayerNorm instead of Batch Renormalization (no batch
statistics). Momentum sweeps are not planned: BRN running statistics average over roughly 100
updates at the default momentum, far shorter than the 400k-step drift.

## Metrics

- **Primary:** mean evaluation return over the 20 evaluations (200 episodes) between 400k and 500k steps, per run.
- **Secondary:** solved rate over the same window; area under the learning curve (0-500k);
  the largest drop in return during the ramp (0-400k), relative to the Fixed run of the same seed.

## Analysis

- **H1:** per-seed paired difference d = Fixed - Drift (CrossQ). H1 is supported if the 95%
  bootstrap confidence interval of the mean of d (10,000 resamples of seeds) lies entirely above 0.
  If it lies entirely below 0, H1 is contradicted; otherwise the result is inconclusive.
- **H2:** per-seed difference-in-differences (Fixed - Drift for CrossQ) minus (Fixed - Drift for
  SAC), same bootstrap rule.
- Learning curves: mean and 95% bootstrap interval over seeds; IQM reported alongside the mean.
- Random-policy reference: `scripts/random_baseline.sh` at fatigue 0 and 0.5.

## Decision rules (fixed in advance)

1. **Fixed condition too hard.** If, after Batch 1, CrossQ under Fixed is not above the random
   policy's return in the final 100k steps for at least 2 of 3 seeds, the level is lowered to 0.4
   for **all** conditions and Batch 1 is rerun before any other main run. Recorded as an amendment.
2. **Adding seeds.** If, after seeds 11-15, the 95% interval for H1 includes 0, seeds 16-18 are
   added to **every** condition of the main experiment (SAC and CrossQ, all three conditions)
   before the mechanism experiment. No other rule adds runs.
3. **Failed runs.** A run that crashes is rerun with the same seed and config. Runs are never
   excluded or repeated because of their performance.
4. **No changes after seeing results.** Metrics, windows, thresholds and conditions above are not
   changed. Any necessary change is added as a dated amendment below, with its reason.

## Batches

| Batch | Runs | Status |
|---|---|---|
| 1 | CrossQ Fixed + Drift, seeds 11-13 (6 runs) | done 8 Nov (all exit 0, ~140 min each; rule 1 passed) |
| 2 | CrossQ Fixed + Drift seeds 14-15; CrossQ no fatigue seeds 11-12 (6 runs); plus pilot `configs/pilot/handreach_crossqln_nofatigue.cfg` (CrossQ with LayerNorm, seed 1) to decide H3 | done 8 Nov (all exit 0, 131-141 min each). H1 after seeds 11-15: inconclusive (`tables/h1_crossq.md`), so decision rule 2 applies |
| 3 | CrossQ no fatigue seeds 13-15; SAC Fixed + Drift seeds 11-13 (9 runs) | planned |
| 4 | SAC Fixed + Drift seeds 14-15; SAC no fatigue seeds 11-15; Walker2d bridge in the MyoSuite software setup (MuJoCo 3.1.2 + patch), 1 seed per algorithm (11 runs) | planned |

Each batch is sized to finish within Kaggle's 12-hour session limit on 2x T4 (CrossQ about 171 min
and SAC about 90 min per Hand Reach run, measured in the pilot).

## Amendments

**Amendment 1 (7 Nov 2026, before any main-experiment run).** Drift also acts as an easy-to-hard
curriculum (full strength first), which may improve final performance and offset a
non-stationarity penalty. H1 is therefore a conservative test: support means the penalty outweighs
any curriculum benefit, while no difference or Drift > Fixed is reported as inconclusive about
non-stationarity, not as evidence against it. H2 and H3 compare conditions that receive the same
curriculum, so they remain interpretable either way. The largest drop in return during the ramp,
relative to the Fixed run of the same seed, is reported as the most direct sign of disruption.
Note on pairing: runs with the same seed share their initial network and first target; target
sequences diverge afterwards because episodes end at different times.
