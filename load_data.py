"""
Load grape datasets for processing
"""
import os
import pandas as pd
import numpy as np
import argparse
import sys

DATASET_DIRECTORY = "/Users/wsolow/Projects/grape_datasets/Frost_Mitigation_Datasets/ColdHardiness/Grapes/Processed/WashingtonState/Prosser/Excel/"

GRAPE_CULTIVARS = ['Aligote', 'Alvarinho', 'Auxerrois', 'Barbera', 'Cabernet Franc', 
                   'Cabernet Sauvignon', 'Chardonnay', 'Chenin Blanc', 'Concord', 
                   'Dolcetto', 'Durif', 'Gewurztraminer', 'Green Veltliner', 'Grenache', 
                   'Lemberger', 'Malbec', 'Melon', 'Merlot', 'Muscat Blanc', 'Nebbiolo', 
                   'Petit Verdot', 'Pinot Blanc', 'Pinot Gris', 'Pinot Noir', 'Riesling', 
                   'Sangiovese', 'Sauvignon Blanc', 'Semillon', 'Syrah', 'Tempranillo', 
                   'Viognier', 'Zinfandel']

PHENO_STAGES = ["Ecodorm", "Budburst/Budbreak", "Full Bloom", "Veraison 50%", "Harvest", "Endodorm"]
PHENOLOGY_INT = {"Ecodorm":0, "Budburst/Budbreak":1, "Full Bloom":2, "Veraison 50%":3, "Harvest":4, "Endodorm":5}

RENAME_COLUMN_MAP = {"SR_WM2":"IRRAD", "P_INCHES":"RAIN", "MEAN_AT":"TEMP", "MAX_AT":"TMAX", "MIN_AT":"TMIN",}

COL_ORDERING = ["DATE", "PHENOLOGY", "TMIN", "TMAX", "TEMP", "RAIN", "IRRAD", "LAT", "LON"]
MPH_TO_MS = 0.44704
IN_TO_CM = 2.54
MJ_TO_J = 1000000 

LAT = 40
LON = -120


def load_and_process_data(cultivar: str):
    """
    Load and process AgAid GitHub repository data
    """
    
    df = pd.read_csv(f"{DATASET_DIRECTORY}"+f'ColdHardiness_Grape_{cultivar}.csv')

    # Remove all grape stages we are not interested in predicting
    df.loc[~df["PHENOLOGY"].isin(PHENO_STAGES),"PHENOLOGY"] = np.nan
    # Get the switch to the dormancy season so we can add endodorm
    for ds in np.argwhere(np.diff(df["DORMANT_SEASON"],prepend=[0]) == 1):
        df.loc[ds, "PHENOLOGY"] = "Endodorm"

    # TODO: With real values
    # For backfilling purposes
    for ed in np.argwhere(df["YEAR_JDAY"]==1):
        df.loc[ed, "PHENOLOGY"] = "Ecodorm"
    # Arbitrarily choose a day for ecodormancy to start, currently november 30th
    for ed in np.argwhere(df["YEAR_JDAY"]==334):
        df.loc[ed, "PHENOLOGY"] = "Ecodorm"

    # Forward fill with non-na values
    df["PHENOLOGY"] = df["PHENOLOGY"].ffill()
    
    # Covert phenology to int values
    for i in range(len(df["PHENOLOGY"])):
        df.loc[i,"PHENOLOGY"] = PHENOLOGY_INT[df["PHENOLOGY"].iloc[i]]
   
    # Drop all columns we don't care about
    df.drop(columns=["AWN_STATION", "SEASON", "SEASON_JDAY", "LTE10", "LTE50", "LTE90", 
              "PREDICTED_LTE10", "PREDICTED_LTE50", "PREDICTED_LTE90", 
             "PREDICTED_BUDBREAK", "MIN_ST2", "ST2", "MAX_ST2", "MIN_ST8", "ST8", "MAX_ST8",
             "SM8_PCNT", "SWP8_KPA", "MSLP_HPA", "LW_UNITY", "ETO", "ETR"],inplace=True)
    
    # Unit conversions
    # Convert MJ to J
    df.loc[:,"SR_WM2"] *= MJ_TO_J

    # Convert inches of rainfall to cm
    df.loc[:,"P_INCHES"] *= IN_TO_CM

    # Convet mph wind speed to m/s
    df.loc[:, ["WS_MPH", "MAX_WS_MPH"]] *= MPH_TO_MS
    
    # Rename columns for compatibility 
    df.rename(columns=RENAME_COLUMN_MAP,inplace=True)

    # Add latitute and longitude columns
    df["LAT"] = LAT
    df["LON"] = LON

    df.drop(columns=["AVG_AT", "MIN_RH", "AVG_RH", "MAX_RH", "MIN_DEWPT", "AVG_DEWPT", 
                     "MAX_DEWPT", "WS_MPH", "MAX_WS_MPH", "WD_DEGREE"], inplace=True)

    df.to_csv('/Users/wsolow/Projects/digital_twin/data/aligote.csv')
    df_list = []

    df_flag = False
    ys = np.argwhere(df["YEAR_JDAY"] == 1).flatten()

    for i in range(len(ys)):
        # Get the slices of each individual year
        if i == len(ys) - 1:
            year_df = df[ys[i]:].copy()
        else:
            year_df = df[ys[i]:ys[i+1]].copy()

        pheno_states = np.unique(year_df["PHENOLOGY"])
        # If there are any nan values throw out the entire year
        if year_df.isnull().any().any():
            df_flag = False
            continue

        # If missing budbreak, flowering, or verasion, throw out year
        
        elif PHENOLOGY_INT["Budburst/Budbreak"] not in pheno_states:
            df_flag = False
            continue

        elif PHENOLOGY_INT["Full Bloom"] not in pheno_states:
            df_flag = False
            continue

        elif PHENOLOGY_INT["Veraison 50%"] not in pheno_states:
            df_flag = False
            continue

        # Otherwise append
        else:
            if not df_flag:
                df_list.append(year_df)
                df_flag = True
            else:
                df_list[-1] = pd.concat([df_list[-1], year_df],axis=0) 

    for yr_df in df_list:
        yr_df.drop(columns=["DORMANT_SEASON", "YEAR_JDAY"],inplace=True)
        yr_df = yr_df[COL_ORDERING]

    return df_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cultivar", type=str, default="Aligote", help='Grape cultivar, see directory for possible cultivars')

    args = parser.parse_args()

    load_and_process_data(args.cultivar)
if __name__ == "__main__":
    main()