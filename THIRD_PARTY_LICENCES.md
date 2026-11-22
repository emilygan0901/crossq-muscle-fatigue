# Third-party code and licences

| Component | Used how | Licence | Source |
|---|---|---|---|
| CrossQ (Bhatt et al., ICLR 2024), including its `sbx` fork of SBX/Stable-Baselines3 | Cloned at commit `a318a266679bec3f9d93c44e029860a6da077c89`. Used unchanged for the reproduction. For the MyoSuite runs, three small edits are applied as `patches/crossq_coursework.patch`. The CrossQ code itself is **not** copied into this repository. | MIT (see the CrossQ repository's `LICENSE` and `NOTICE`, which also cover the ported Stable-Baselines and OpenAI Baselines code) | https://github.com/adityab/CrossQ |
| MyoSuite 2.8.4 | Installed from PyPI. `src/fatigue_drift.py` wraps its `myoFatiHandReachRandom-v0` task and sets its fatigue state. No MyoSuite code is copied. | Apache 2.0 | https://github.com/MyoHub/myosuite |
| MuJoCo, JAX, Flax, Gymnasium and other packages | Installed from PyPI at the versions in `environment/` | Their own licences (MuJoCo Apache 2.0, JAX Apache 2.0, Gymnasium MIT) | PyPI |

`patches/crossq_coursework.patch` is a diff against CrossQ's `train.py` and therefore falls under
CrossQ's MIT licence. The rest of this repository (wrapper, configs, scripts, tests, analysis and
results) is the author's own work for BIOE70077.
