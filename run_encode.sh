CUR_EXPERIMENT_DIR=$(dirname $(readlink -f "$0"))
CODE_DIR="/path/to/anchordx_ngs"

cd $CODE_DIR
echo "running on $CUR_EXPERIMENT_DIR"

python -m run auto_encoder_encode 2>&1 --experiment_dir $CUR_EXPERIMENT_DIR --n_splits $1 $2