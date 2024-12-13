#!/bin/bash
#SBATCH -J iterative
#SBATCH -o output/iterative.out
#SBATCH -e output/iterative.err
#SBATCH -p eecs,gpu,share
#SBATCH -t 1-00:00:00
#SBATCH --mem=10G

python3 iterative_optimizer.py
