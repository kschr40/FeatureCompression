#!/bin/bash

# Ensure you're using bash
#datasets=("magic" "spambase" "statlog" "cycle" "concrete" "superconductor") #"gas"
#types=("binary" "binary" "multiclass" "regression" "regression" "regression" ) #"multiclass"
timestamp=$(date +%s)

# Loop through each dataset and its corresponding type
python3 hyperparameter_tuning.py --dataset california --scratch /scratch/tmp/n_herr03/FeatureCompression --result_folder /scratch/tmp/n_herr03/FeatureCompression > /scratch/tmp/n_herr03/FeatureCompression/outdata/california-$timestamp.out
