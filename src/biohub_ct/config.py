from __future__ import annotations

from pathlib import Path

DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
MAX_MATCH_DISTANCE_MICRONS = 7.0
ADJUSTMENT_ALPHA = 0.1
SCORE_DIVISION_WEIGHT = 0.1

SUBMISSION_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]

KAGGLE_COMPETITION_DIRS = [
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
]

