#!/bin/bash
#SBATCH -J exp_4
#SBATCH -o output/exp_4.out
#SBATCH -e output/exp_4.err
#SBATCH -p eecs,share
#SBATCH -t 5-00:00:00

python3 bayesian_optimizer_multi.py --seed 0 --cultivar "Multi"
python3 bayesian_optimizer_multi.py --seed 1 --cultivar "Multi" 
