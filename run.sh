#!/bin/bash

for cultivar in "Aligote" "Alvarinho" "Auxerrois" "Barbera" "Cabernet Franc" "Cabernet Sauvignon" "Chardonnay" "Chenin Blanc" "Concord" "Dolcetto" "Durif" "Gewurztraminer" "Green Veltliner" "Grenache" "Lemberger" "Malbec" "Melon" "Merlot" "Muscat Blanc" "Nebbiolo" "Petit Verdot" "Pinot Blanc" "Pinot Gris" "Pinot Noir" "Riesling" "Sangiovese" "Sauvignon Blanc" "Semillon" "Syrah" "Tempranillo" "Viognier" "Zinfandel"
do
    echo $cultivar
    python3 optimizer.py --cultivar $cultivar &
done