#!/bin/bash
#SBATCH -J single
#SBATCH -o output/single.out
#SBATCH -e output/single.err
#SBATCH -p eecs,gpu,share
#SBATCH -t 1-00:00:00

python3 bayesian_optimizer.py --cultivar "Zinfandel"
