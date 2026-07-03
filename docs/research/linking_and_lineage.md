# Linking And Lineage

## Current Baseline

The implemented linker is adjacent-frame greedy assignment with physical distance gates. It enforces one parent per child and one child per source unless division mode is explicitly enabled.

## Next Linkers

- LAP frame-to-frame assignment using SciPy when available.
- Two-pass motion relinking for missed local assignments.
- One-frame gap closing after metric probes confirm how non-adjacent edges score.
- Division proposals with strict thresholds on daughter distance, intensity, and absence of a better single continuation.
- Optional ILP/Motile integration if dependencies and runtime fit Kaggle.

## Metric Implications

Edges dominate the final score. A predicted edge touching annotated GT in/out regions becomes a false positive if it does not match a GT edge. Edges entirely in unannotated regions are ignored, but their nodes still contribute to total predicted node count.

