#!/bin/bash

#SBATCH --export=NONE
#SBATCH --partition=d0giesek,gpu4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --job-name=featurecomp_hyperparameter_ext_wisconsin
#SBATCH --output=/scratch/tmp/k_schr40/FeatureCompression/featurecompression-reg_hyperparam_ext_wisconsin.out
#SBATCH --error=/scratch/tmp/k_schr40/FeatureCompression/featurecompression-reg_hyperparam_ext_wisconsin.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=k_schr40@uni-muenster.de
#SBATCH --mem=50GB

partition=d0giesek

module purge
ml palma/2023a
ml GCC/12.3.0
ml OpenMPI/4.1.5
ml CUDA/12.2.2
ml PyTorch/2.1.2-CUDA-12.1.
pip install —upgrade pip

pip install torch
pip install pandas
pip install random
pip install tqdm
pip install argparse
pip install os
pip install numpy
pip install openml

cd /home/k/k_schr40/

cd FeatureCompression
./run_extending_hyperparameter.sh
