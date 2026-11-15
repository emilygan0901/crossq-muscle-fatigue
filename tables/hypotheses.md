| Test | Algorithm | Comparison | Seeds | Mean difference [95% bootstrap CI] | 95% t-interval | MDE | Verdict |
|---|---|---|---|---|---|---|---|
| Check | CrossQ | Cost of fatigue (No fatigue - Fixed) | 8 | 119.3 [46.7, 203.0] | [17.5, 221.1] | 140 | check (not a hypothesis) |
| H1 | CrossQ | Drift penalty (Fixed - Drift) | 8 | -32.9 [-103.9, 28.5] | [-119.6, 53.7] | 119 | inconclusive |
| H4 | CrossQ | Sudden penalty (Fixed - Sudden, 255k-350k) | 5 | 31.7 [-7.6, 65.0] | [-25.3, 88.7] | 76 | inconclusive |
| Check | SAC | Cost of fatigue (No fatigue - Fixed) | 5 | -2.8 [-34.4, 23.9] | [-51.0, 45.3] | 64 | check (not a hypothesis) |
| H2 part | SAC | Drift penalty (Fixed - Drift) | 5 | -22.6 [-61.2, 25.1] | [-89.9, 44.8] | 90 | not testable: SAC did not learn the task (Amendment 3) |
| H2 | both | Drift penalty, CrossQ minus SAC | 5 | 16.6 [-76.8, 91.2] | [-120.5, 153.8] | 184 | not testable: SAC did not learn the task (Amendment 3) |

H3, H6 and H7 (role of batch normalisation) were not testable: neither variant without batch normalisation learned the task (LayerNorm pilot; no-BN pilot, Amendment 3).

Positive = the second condition scores lower (a penalty). Verdict rule (pre-registered): supported if the 95% CI is entirely above 0, contradicted if entirely below 0. MDE = minimum detectable effect (80% power, paired t-test, alpha 0.05).
