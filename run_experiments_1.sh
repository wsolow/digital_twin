#!/bin/bash
#SBATCH -J exp_1
#SBATCH -o output/exp_1.out
#SBATCH -e output/exp_1.err
#SBATCH -p eecs,share
#SBATCH -t 4-00:00:00

python3 bayesian_optimizer.py --seed 1 --cultivar "Aligote"
python3 bayesian_optimizer.py --seed 1 --cultivar "Alvarinho" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Auxerrois" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Barbera" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Cabernet_Franc" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Cabernet_Sauvignon" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Chardonnay" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Chenin_Blanc" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Concord" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Dolcetto" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Durif" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Gewurztraminer" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Green_Veltliner" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Grenache" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Lemberger" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Malbec" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Melon" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Merlot" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Muscat_Blanc" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Nebbiolo" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Petit_Verdot" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Pinot_Blanc" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Pinot_Gris" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Pinot_Noir" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Riesling" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Sangiovese" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Sauvignon_Blanc" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Semillon" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Syrah" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Tempranillo" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Viognier" 
python3 bayesian_optimizer.py --seed 1 --cultivar "Zinfandel"
