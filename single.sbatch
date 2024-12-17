#!/bin/bash
#SBATCH -J single
#SBATCH -o output/single.out
#SBATCH -e output/single.err
#SBATCH -p eecs,gpu,share
#SBATCH -t 1-00:00:00

CULTIVAR="$1"
python3 eval.py --cultivar "$CULTIVAR"
