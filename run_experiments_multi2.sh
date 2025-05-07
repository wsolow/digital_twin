#!/bin/bash
#SBATCH -J exp_multi_2
#SBATCH -o output/exp_multi_2.out
#SBATCH -e output/exp_multi_2.err
#SBATCH -p eecs,share
#SBATCH -t 3-00:00:00 

python3 bayesian_optimizer_multi.py --seed 2 --cultivar "Multi" 
 