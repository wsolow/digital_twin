#!/bin/bash8
for cultivar in "Aligote" "Alvarinho" "Auxerrois" "Barbera" "Cabernet_Franc" "Cabernet_Sauvignon" "Chardonnay" "Chenin_Blanc" "Concord" "Dolcetto" "Durif" "Gewurztraminer" "Green_Veltliner" "Grenache" "Lemberger" "Malbec" "Melon" "Merlot" "Muscat_Blanc" "Nebbiolo" "Petit_Verdot" "Pinot_Blanc" "Pinot_Gris" "Pinot_Noir" "Riesling" "Sangiovese" "Sauvignon_Blanc" "Semillon" "Syrah" "Tempranillo" "Viognier" "Zinfandel"
do
    echo $cultivar
    python3 bayesian_optimizer.py --cultivar $cultivar --seed 0 & 
done