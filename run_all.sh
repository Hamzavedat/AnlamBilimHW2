#!/bin/bash
set -e

source venv/bin/activate
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python src/collect_data.py
python src/extract_features.py
python src/predict.py
python src/visualize.py

echo "Pipeline finished."
echo "Outputs are in runs/hw2_full_seed42/"
