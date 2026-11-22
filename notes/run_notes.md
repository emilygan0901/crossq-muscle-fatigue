# Run notes

A chronological record of every batch of runs: what it ran, why, where it ran, and what did not
work. Dates are UTC run start dates from the run folder names. The planned design and its
amendments are in `HYPOTHESIS.md`. Every run that produced results has a folder in `results/raw/`
and a row in `results/run_log.md`.

All GPU runs ran on Kaggle (NVIDIA T4, 2 GPUs per session, one run per GPU), started with
"Save & Run All (Commit)". Each notebook has four cells: setup (`environment/setup_kaggle.sh` for
the MyoSuite batches), checks, the batch command, and a cell that zips the new run folders.

## Kaggle notebooks

| Batch | Kaggle notebook and version | Results zip |
|---|---|---|
| Reproduction (Walker2d) | [[notebook name, version]] | [[file name]] |
| Elbow pilots | [[notebook name, version]] | `pilot_results` |
| Hand Reach pilot | [[notebook name, version]] | [[file name]] |
| Batch 1 | [[notebook name, version]] | [[file name]] |
| Batch 2 | [[notebook name, version]] | [[file name]] |
| Batch 3 | [[notebook name, version]] | [[file name]] |
| Batch 4, first attempt (no GPU) | [[notebook name, version]] | `results_5.zip`, not merged (all runs crashed) |
| Batch 4 | [[notebook name, version]] | `results_6.zip` |
| Batch 5 | [[notebook name, version]] | `results_7.zip` |
| Batch 6 | [[notebook name, version]] | `results_8.zip` |
| Batch 9 | [[notebook name, version]] | `results_9.zip` |

## Chronology

| Date | Batch | Runs | Purpose and outcome |
|---|---|---|---|
| 6-7 Nov | Reproduction | CrossQ and SAC on Walker2d-v4, seeds 1-5, 500k steps, authors' code unchanged (`configs/repro/`) | Scaled reproduction of CrossQ's Fig. 5. Qualitative result reproduced (IQM 6326 vs 3916 at 500k). `tables/repro_walker2d.md` |
| 6 Nov | Elbow pilots | CrossQ and SAC (twice) on the MyoSuite elbow pose task, 300k steps, seed 1 | Rejected the task: both learners reached 95% of their best return within about 6,500 steps (ceiling). `results/pilot_elbow/` |
| 7 Nov | Hand Reach pilot | CrossQ and SAC, No fatigue and Drift (0 to 0.6), seed 1 (`configs/pilot/`) | Hand Reach is learnable in 500k steps without a ceiling, drift lowers performance, timings measured. Showed that a Fixed control was needed to separate a changing body from a weaker one |
| 7 Nov | Pre-registration | `HYPOTHESIS.md` committed before any main run | |
| 7-8 Nov | Batch 1 | CrossQ Fixed + Drift, seeds 11-13 | Decision rule 1 (Fixed above random) passed |
| 8 Nov | Batch 2 | CrossQ Fixed + Drift seeds 14-15, No fatigue seeds 11-12, LayerNorm pilot | H1 inconclusive after 5 seeds, so rule 2 (add seeds 16-18) applies. LayerNorm pilot did not learn (see below) |
| 9 Nov | Batch 3 | CrossQ No fatigue seeds 13-15, SAC all conditions seeds 11-12 | SAC solved no episodes |
| 14 Nov | Batch 4 | CrossQ Sudden seeds 11-15, Walker2d bridge runs (CrossQ, SAC) | First attempt crashed (no GPU, see below), rerun unchanged |
| 14 Nov | Batch 5 | CrossQ Fixed + Drift seeds 16-18, pilot of CrossQ without batch normalisation | Pilot failed its pre-registered pass rule (Amendment 3) |
| 15 Nov | Batch 6 | CrossQ No fatigue seeds 16-18, SAC all conditions seeds 13-15 | All planned experiments complete |
| 22 Nov | Batch 9 | CrossQ No fatigue + Fixed seeds 11-13 with a final fatigue sweep (`configs/sweep/`) | Amendment 4, exploratory, written before the runs. `tables/sweep_crossq.md` |

Batches 7-8 (CrossQ without batch normalisation, all conditions) were not run because their pilot
failed.

## What did not work

- **Elbow task too easy.** Both algorithms hit the ceiling within the first few thousand steps, so
  no condition could show an effect. Replaced by Hand Reach.
- **MyoSuite's built-in fatigue too weak.** Even maximal activation of every muscle fatigues a muscle
  by at most 1.7% within a default 2-s episode, and the state resets every episode
  (`scripts/check_builtin_fatigue.py`, `results/builtin_fatigue.csv`). Replaced by our own schedule
  (`src/fatigue_drift.py`), which sets the fatigued fraction from the training step.
- **First SAC elbow pilot used the wrong flags.** It learned far more slowly than it should have
  (return 29 over the first 25k steps, against 607 after the fix) and was rerun with the corrected
  flags. Kept in `results/pilot_elbow/sac_seed1_first/`, not used.
- **Lost Walker2d run.** [[describe what happened to the lost reproduction run and how it was
  replaced]]
- **LayerNorm pilot, a design flaw.** CrossQ's `-crossq_style 1` switch also sets tau = 1, which
  removes the target network. The pilot swapped Batch Renormalization for LayerNorm but kept that
  switch, so it had neither stabiliser. It rose to a return of 311 and then fell to 226 (0%
  solved), the pattern of an unstable critic. This tests a combination that was unlikely to work and
  says nothing about LayerNorm itself. The flaw came from the AI-assisted setup (see the AI-use
  section of the README) and is recorded in Amendment 3.
- **CrossQ without batch normalisation also failed.** Set up correctly (target network restored),
  it still solved no episodes (return 53). With SAC also unable to learn, the role of batch
  normalisation could not be isolated (H2, H3, H5, H6, H7 not testable).
- **SAC never solved Hand Reach** within 500k steps. Its returns rose slowly and levelled off
  (seed 12: 112 over 150k-200k, 115 over 450k-500k). The same SAC settings reproduced Walker2d, so
  the setup works. Further SAC runs were stopped (Amendment 3).
- **Batch 4 first attempt without a GPU.** The notebook started without the GPU accelerator
  switched on, and all 7 runs crashed at start-up before any training. They were rerun unchanged
  under decision rule 3. `scripts/run_batch.sh` now stops with an error if no GPU is found.
- **Seed pairing reduced noise less than expected.** GPU computation is not bit-for-bit
  deterministic, so runs with the same seed are identical only until learning starts and differ
  from 10,000 steps on (`tables/run_to_run_variation.md`).

## Software setup fixes

- MuJoCo is overridden to 3.1.2 in the MyoSuite environment because CrossQ pins 2.3.7, which no
  gymnasium-based MyoSuite release supports. The Walker2d bridge runs check that this change does
  not alter the reproduction pattern (`tables/bridge_walker2d.md`).
- MyoSuite pulls in torch 2.0.1, whose CUDA 11 libraries overwrite files JAX needs, so the CUDA 12
  solver and cuDNN are reinstalled last (`environment/setup_kaggle.sh`).
- The reproduction environment pins cuDNN 8.9.7.29 in addition to the authors' requirements.

## Local checks (not results)

The code was also run on a CPU in the AI assistant's workspace, only to check that it works:
short runs of a few thousand steps (for example of the sweep code before Batch 9), the unit tests
in `tests/`, and probes such as `scripts/check_activation_saturation.py`. None of these short runs
is reported as a result. Every reported learning result comes from the Kaggle runs above.
