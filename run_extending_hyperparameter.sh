#!/bin/bash

# Ensure you're using bash
#datasets=("wine_quality" "california" "BNG(wisconsin)" "fried" "NewFuelCar" "superconduct") #"gas"
#num_bits=(2 3 4 5 6 7 8) #"BNG(wisconsin)")
#types=("binary" "binary" "multiclass" "regression" "regression" "regression" ) #"multiclass"
timestamp=$(date +%s)

# Loop through each dataset and its corresponding type
#for dataset in $datasets; do
dataset="BNG(wisconsin)"

python3 extend_hyperparameter_tuning.py --dataset $dataset --n_bits_min 4 --n_bits_max 4 --n_steps 100 --scratch /scratch/tmp/k_schr40/FeatureCompression --result_folder /scratch/tmp/k_schr40/FeatureCompression/results > /scratch/tmp/k_schr40/FeatureCompression/outdata/$dataset-$timestamp.out

