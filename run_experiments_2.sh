#!/bin/bash
#SBATCH -J exp_2
#SBATCH -o output/exp_2.out
#SBATCH -e output/exp_2.err
#SBATCH -p eecs,share
#SBATCH -t 4-00:00:00

python3 bayesian_optimizer.py --seed 2 --cultivar "Aligote"
python3 bayesian_optimizer.py --seed 2 --cultivar "Alvarinho" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Auxerrois" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Barbera" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Cabernet_Franc" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Cabernet_Sauvignon" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Chardonnay" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Chenin_Blanc" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Concord" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Dolcetto" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Durif" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Gewurztraminer" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Green_Veltliner" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Grenache" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Lemberger" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Malbec" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Melon" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Merlot" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Muscat_Blanc" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Nebbiolo" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Petit_Verdot" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Pinot_Blanc" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Pinot_Gris" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Pinot_Noir" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Riesling" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Sangiovese" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Sauvignon_Blanc" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Semillon" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Syrah" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Tempranillo" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Viognier" 
python3 bayesian_optimizer.py --seed 2 --cultivar "Zinfandel"
