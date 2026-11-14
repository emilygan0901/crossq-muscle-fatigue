| Condition | Seeds | 200-250k (before) | 255k (first after) | 255-300k | 300-350k | 400-500k |
|---|---|---|---|---|---|---|
| Sudden at 250k | 5 | 378 | 371 | 402 | 391 | 475 |
| Fixed 0.5 | 5 | 402 | 372 | 425 | 434 | 490 |
| No fatigue | 5 | 440 | 524 | 481 | 510 | 572 |

CrossQ mean evaluation return per window (mean over seeds). Sudden and No fatigue are identical conditions until 250k, so their difference before the jump shows run-to-run variation alone (GPU computation is not bit-for-bit deterministic, so same-seed runs diverge once training starts).
