#!/bin/bash
#SBATCH -J exp_0
#SBATCH -o output/exp_0.out
#SBATCH -e output/exp_0.err
#SBATCH -p eecs,share
#SBATCH -t 4-00:00:00

python3 bayesian_optimizer.py --seed 0 --cultivar "Aligote"
python3 bayesian_optimizer.py --seed 0 --cultivar "Alvarinho" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Auxerrois" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Barbera" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Cabernet_Franc" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Cabernet_Sauvignon" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Chardonnay" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Chenin_Blanc" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Concord" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Dolcetto" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Durif" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Gewurztraminer" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Green_Veltliner" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Grenache" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Lemberger" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Malbec" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Melon" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Merlot" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Muscat_Blanc" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Nebbiolo" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Petit_Verdot" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Pinot_Blanc" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Pinot_Gris" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Pinot_Noir" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Riesling" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Sangiovese" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Sauvignon_Blanc" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Semillon" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Syrah" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Tempranillo" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Viognier" 
python3 bayesian_optimizer.py --seed 0 --cultivar "Zinfandel"
