#!/bin/bash

# Define an array of input parameters to vary
cultivars=("Aligote" "Alvarinho" "Auxerrois" "Barbera" "Cabernet_Franc" 
                   "Cabernet_Sauvignon" "Chardonnay" "Chenin_Blanc" "Concord" 
                   "Dolcetto" "Durif" "Gewurztraminer" "Green_Veltliner" "Grenache" 
                   "Lemberger" "Malbec" "Melon" "Merlot" "Muscat_Blanc" "Nebbiolo" 
                   "Petit_Verdot" "Pinot_Blanc" "Pinot_Gris" "Pinot_Noir" "Riesling" 
                   "Sangiovese" "Sauvignon_Blanc" "Semillon" "Tempranillo" 
                   "Viognier" "Zinfandel") # Replace with your actual inputs

# Loop through the input values
for cultivar in "${cultivars[@]}"; do
    # Submit the job with the current input as an argument
    #sbatch single.sbatch "$cultivar"
    python3 to_yaml_batch.py --cultivar "$cultivar"
done

echo "Submitted jobs with varying inputs: ${cultivars[*]}"
