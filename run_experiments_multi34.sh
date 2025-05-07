#!/bin/bash
#SBATCH -J multi34
#SBATCH -o output/multi34.out
#SBATCH -e output/multi34.err
#SBATCH -p eecs,share
#SBATCH -t 5-00:00:00

python3 bayesian_optimizer_multi.py --seed 3 --cultivar "Multi" 
python3 bayesian_optimizer_multi.py --seed 4 --cultivar "Multi" 