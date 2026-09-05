# Source this file from the server shell.
export BIOHUB_PROJECT=/workspace/biohub-cell-tracking
export CELLMOT_DATA_DIR="$BIOHUB_PROJECT/data/train"
export KAGGLEHUB_CACHE="$BIOHUB_PROJECT/cache/kagglehub"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
source "$BIOHUB_PROJECT/.venv/bin/activate"
cd "$BIOHUB_PROJECT"
