#!/bin/bash

# Ensure you're using bash
#datasets=("wine_quality" "california" "BNG(wisconsin)" "fried" "NewFuelCar" "superconduct") #"gas"
datasets=("superconduct") #("wine_quality" "california" "fried") #"BNG(wisconsin)")
#types=("binary" "binary" "multiclass" "regression" "regression" "regression" ) #"multiclass"
timestamp=$(date +%s)

# Loop through each dataset and its corresponding type
#for dataset in $datasets; do
for dataset in "${datasets[@]}"; do
python3 evaluate_different_bits.py --dataset $dataset --n_bits_min 2 --n_bits_max 8 --scratch /scratch/tmp/k_schr40/FeatureCompression --result_folder /scratch/tmp/k_schr40/FeatureCompression/results > /scratch/tmp/k_schr40/FeatureCompression/outdata/$dataset-different-bits-$timestamp.out
done
