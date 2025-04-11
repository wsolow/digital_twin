#!/bin/bash
#SBATCH -J exp_3
#SBATCH -o output/exp_3.out
#SBATCH -e output/exp_3.err
#SBATCH -p eecs,share
#SBATCH -t 4-00:00:00

python3 bayesian_optimizer.py --seed 3 --cultivar "Aligote"
python3 bayesian_optimizer.py --seed 3 --cultivar "Alvarinho" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Auxerrois" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Barbera" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Cabernet_Franc" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Cabernet_Sauvignon" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Chardonnay" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Chenin_Blanc" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Concord" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Dolcetto" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Durif" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Gewurztraminer" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Green_Veltliner" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Grenache" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Lemberger" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Malbec" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Melon" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Merlot" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Muscat_Blanc" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Nebbiolo" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Petit_Verdot" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Pinot_Blanc" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Pinot_Gris" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Pinot_Noir" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Riesling" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Sangiovese" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Sauvignon_Blanc" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Semillon" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Syrah" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Tempranillo" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Viognier" 
python3 bayesian_optimizer.py --seed 3 --cultivar "Zinfandel"
