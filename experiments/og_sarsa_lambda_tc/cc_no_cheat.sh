#!/bin/bash
#SBATCH --account=def-jpineau
#SBATCH --cpus-per-task=25
#SBATCH --mail-user=ruo.tao@mail.mcgill.ca
#SBATCH --mail-type=ALL
#SBATCH --mem=3000M
#SBATCH --time=03-00:00

source ~/scratch/generalization-rl/venv/bin/activate

python run_mp.py --no-cheat