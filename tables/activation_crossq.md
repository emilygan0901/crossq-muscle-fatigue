Mean over seeds (11-13) of each policy's mean over the episodes at each test level (CPU replay of the final-policy sweep).

| Training condition | Measure | 0 | 0.2 | 0.4 | 0.5 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|---|
| Trained without fatigue | commanded drive | 0.263 | 0.262 | 0.263 | 0.269 | 0.277 | 0.310 |
| Trained without fatigue | actual activation | 0.283 | 0.267 | 0.227 | 0.202 | 0.173 | 0.101 |
| Trained without fatigue | fraction of muscles at capacity | 0.000 | 0.148 | 0.219 | 0.247 | 0.275 | 0.402 |
| Trained without fatigue | final distance (m) | 0.068 | 0.068 | 0.068 | 0.071 | 0.076 | 0.106 |
| Trained without fatigue | steps to solve (solved episodes) | 10 | 11 | 14 | 18 | 19 | 30 |
| Trained without fatigue | solved fraction | 0.44 | 0.44 | 0.45 | 0.43 | 0.39 | 0.23 |
| Trained at fatigue 0.5 | commanded drive | 0.205 | 0.203 | 0.202 | 0.204 | 0.205 | 0.275 |
| Trained at fatigue 0.5 | actual activation | 0.243 | 0.224 | 0.188 | 0.167 | 0.142 | 0.094 |
| Trained at fatigue 0.5 | fraction of muscles at capacity | 0.001 | 0.108 | 0.163 | 0.188 | 0.213 | 0.364 |
| Trained at fatigue 0.5 | final distance (m) | 0.078 | 0.078 | 0.077 | 0.077 | 0.077 | 0.116 |
| Trained at fatigue 0.5 | steps to solve (solved episodes) | 15 | 14 | 14 | 17 | 18 | 31 |
| Trained at fatigue 0.5 | solved fraction | 0.34 | 0.35 | 0.37 | 0.37 | 0.35 | 0.22 |

CPU replay check: mean absolute difference from the GPU sweep returns 0.3 (max 2.2) over 36 policy-level pairs.
Exploratory (Amendment 4 policies, 3 seeds per training condition).
