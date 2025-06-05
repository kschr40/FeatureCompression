#!/bin/bash

# Ensure you're using bash
#datasets=("wine_quality" "california" "BNG(wisconsin)" "fried" "NewFuelCar" "superconduct") #"gas"
num_bits=(2 3 4 5 6 7 8) #"BNG(wisconsin)")
#types=("binary" "binary" "multiclass" "regression" "regression" "regression" ) #"multiclass"
timestamp=$(date +%s)

# Loop through each dataset and its corresponding type
#for dataset in $datasets; do
dataset="fried"
for n_bits in "${num_bits[@]}"; do
python3 hyperparameter_tuning.py --only_additional --dataset $dataset --n_bits $n_bits --n_steps 100 --scratch /scratch/tmp/k_schr40/FeatureCompression --result_folder /scratch/tmp/k_schr40/FeatureCompression/results > /scratch/tmp/k_schr40/FeatureCompression/outdata/$dataset-$n_bits-$timestamp.out
done
