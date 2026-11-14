| Algorithm | Condition | Seeds | Final return | IQM | Solved fraction | AUC (mean return 0-500k) |
|---|---|---|---|---|---|---|
| CrossQ | No fatigue | 5 | 572 [543, 591] | 583 | 0.51 [0.43, 0.56] | 416 [381, 445] |
| CrossQ | Fixed 0.5 | 5 | 490 [423, 545] | 504 | 0.36 [0.22, 0.48] | 371 [329, 409] |
| CrossQ | Drift 0 to 0.5 | 5 | 496 [437, 546] | 506 | 0.40 [0.27, 0.50] | 387 [334, 436] |
| CrossQ | Sudden at 250k | 5 | 475 [444, 502] | 475 | 0.35 [0.29, 0.41] | 353 [327, 380] |
| SAC | No fatigue | 2 | 94 | 94 | 0.00 | 71 |
| SAC | Fixed 0.5 | 2 | 111 | 111 | 0.00 | 60 |
| SAC | Drift 0 to 0.5 | 2 | 126 | 126 | 0.00 | 96 |

Mean [95% bootstrap CI over seeds]; no CI with fewer than 3 seeds. Final = 20 evaluations (200 episodes) after 400k steps. Solved = MyoSuite 'solved' at episode end. IQM over seeds (equals the mean below 4 seeds).
Random policy: -49.3 at fatigue 0, -49.5 at fatigue 0.5.
