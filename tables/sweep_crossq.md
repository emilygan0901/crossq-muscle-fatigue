| Training condition | Seeds | 0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|---|
| Trained without fatigue, return | 3 | 559 | 559 | 546 | 525 | 487 | 308 |
| Trained without fatigue, solved | 3 | 0.45 | 0.44 | 0.45 | 0.43 | 0.38 | 0.23 |
| Trained at fatigue 0.5, return | 3 | 482 | 490 | 491 | 486 | 475 | 263 |
| Trained at fatigue 0.5, solved | 3 | 0.31 | 0.35 | 0.37 | 0.37 | 0.34 | 0.22 |

Mean over seeds of each policy's mean over 50 episodes per level (same targets at every level).

| Quantity | Seeds | Per seed | Mean [95% bootstrap CI] | 95% t-interval |
|---|---|---|---|---|
| E1 at 0.5: No-fatigue-trained, test level 0 minus 0.5 (physical cost) | 3 | 37, 65, 0 | 34.0 [-0.2, 65.0] | [-47.2, 115.2] |
| E2: Fixed-trained minus No-fatigue-trained, test level 0.5 | 3 | -108, -4, -4 | -38.5 [-108.0, -3.5] | [-188.1, 111.0] |
| E3: No-fatigue-trained minus Fixed-trained, test level 0 | 3 | 133, 75, 20 | 76.2 [20.2, 133.1] | [-64.1, 216.5] |

E1: No-fatigue-trained return falls from 559 at level 0 to 308 at level 0.8.
E4 (Trained without fatigue): solved fraction never falls below half its level-0 value.
E4 (Trained at fatigue 0.5): solved fraction never falls below half its level-0 value.

All quantities are exploratory (Amendment 4). With 3 seeds the bootstrap CI is unreliable, so the t-interval is the main interval.
