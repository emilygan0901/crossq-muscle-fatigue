| Algorithm | Setup | Seeds | Mean return 200-250k | Mean return 450-500k | Minutes per run |
|---|---|---|---|---|---|
| CrossQ | Reproduction (MuJoCo 2.3.7) | 5 | 5312 (range 4621-6431) | 6109 (range 5490-6866) | 110-165 |
| CrossQ | Bridge (MuJoCo 3.1.2 + patch) | 1 | 5732 | 5312 | 113-113 |
| SAC | Reproduction (MuJoCo 2.3.7) | 5 | 2290 (range 2122-2629) | 3770 (range 3313-4564) | 48-49 |
| SAC | Bridge (MuJoCo 3.1.2 + patch) | 1 | 2006 | 3828 | 49-49 |

One evaluation episode every 5,000 steps (as in the reproduction); window means over evaluations.
