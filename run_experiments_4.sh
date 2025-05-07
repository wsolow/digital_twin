#!/bin/bash
#SBATCH -J exp_4
#SBATCH -o output/exp_4.out
#SBATCH -e output/exp_4.err
#SBATCH -p eecs,share
#SBATCH -t 6-00:00:00

python3 bayesian_optimizer.py --seed 4 --cultivar "Aligote"
python3 bayesian_optimizer.py --seed 4 --cultivar "Alvarinho" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Auxerrois" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Barbera" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Cabernet_Franc" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Cabernet_Sauvignon" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Chardonnay" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Chenin_Blanc" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Concord" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Durif" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Gewurztraminer" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Green_Veltliner" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Grenache" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Lemberger" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Malbec" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Melon" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Merlot" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Muscat_Blanc" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Nebbiolo" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Petit_Verdot" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Pinot_Blanc" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Pinot_Gris" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Pinot_Noir" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Riesling" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Sangiovese" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Sauvignon_Blanc" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Semillon" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Tempranillo" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Viognier" 
python3 bayesian_optimizer.py --seed 4 --cultivar "Zinfandel"
