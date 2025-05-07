#!/bin/bash
#SBATCH -J multi01
#SBATCH -o output/multi01.out
#SBATCH -e output/multi01.err
#SBATCH -p eecs,share
#SBATCH -t 5-00:00:00

python3 bayesian_optimizer_multi.py --seed 0 --cultivar "Multi"
python3 bayesian_optimizer_multi.py --seed 1 --cultivar "Multi" 
