# Decision 0001: Dependency-Light Core

The verified core path uses stdlib plus NumPy.

Reason: the current local environment lacks pandas, scipy, scikit-image, zarr, and networkx. A metadata-only Zarr fallback and internal graph schema let tests and smoke submission run immediately while keeping richer Kaggle/local dependencies optional.

Consequence: real GEFF reading and official metric execution require optional packages. The code must fail clearly or degrade gracefully when those packages are missing.

