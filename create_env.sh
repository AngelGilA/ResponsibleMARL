#!/bin/bash
#Set job requirements
#SBATCH --job-name="create_env"
#SBATCH -t 00:20:00
#SBATCH --output=init_env_%j.out

module load 2022
module load Anaconda3/2022.05

echo "Start updating conda"
conda init
conda update conda
echo "Create MARL2023paper_env"
time conda env create -f environment.yml
conda activate MARL2023paper_env
