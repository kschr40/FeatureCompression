#!/bin/bash

#SBATCH --export=NONE
#SBATCH --partition=d0giesek
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=168:00:00
#SBATCH --job-name=featurecomp
#SBATCH --output=/scratch/tmp/n_herr03/featurecompression-reg.out
#SBATCH --error=/scratch/tmp/n_herr03/featurecompression-reg.error
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n_herr03@uni-muenster.de
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

cd /home/n/n_herr03/

cd FeatureCompression
./run.sh
