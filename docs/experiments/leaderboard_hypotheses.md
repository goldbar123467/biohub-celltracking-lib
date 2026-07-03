# Leaderboard Hypotheses

| Hypothesis | Metric rationale | Implementation | Expected risk | Validation result | Keep/Drop |
| --- | --- | --- | --- | --- | --- |
| Tune node density before model complexity | Node-count penalty can dominate | Sweep threshold/NMS radius | Under-detect edges | Not run | TBD |
| One-frame gap recovery helps | More edge TP after missed detection | Add ablation flag | Non-adjacent edge mismatch | Not run | TBD |
| Conservative divisions are positive EV | Division term low weight but can break ties | Strict fork proposals | Annotated-region FP | Not run | TBD |
| Learned detector beats classical peaks | Higher recall in dense/noisy frames | Sparse-label 3D heatmap | Dense-label overfit | Not run | TBD |
| Transformer linker improves edges | Edge score dominates final metric | Candidate edge scorer | Runtime and training complexity | Not run | TBD |

