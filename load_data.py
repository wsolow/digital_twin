"""
Load grape datasets for processing
"""
import os
import pandas as pd
import numpy as np
import argparse
import sys
import math
from pathlib import Path

DATASET_DIRECTORY = f"{Path(os.getcwd()).parent.absolute()}/grape_datasets/"

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

def load_and_process_data_dormant(cultivar: str):
    """
    Load and process AgAid GitHub repository data
    """
    
    df = pd.read_csv(f"{DATASET_DIRECTORY}"+f'ColdHardiness_Grape_{cultivar}.csv')

    # Remove all grape stages we are not interested in predicting
    df.loc[~df["PHENOLOGY"].isin(PHENO_STAGES),"PHENOLOGY"] = np.nan

    '''# Get the switch to the dormancy season so we can add endodorm
    for ds in np.argwhere(np.diff(df["DORMANT_SEASON"],prepend=[0]) == 1):
        df.loc[ds, "PHENOLOGY"] = PHENOLOGY_INT["Endodorm"]

    # TODO: With real values
    # For backfilling purposes
    for ed in np.argwhere(df["YEAR_JDAY"]==1):
        df.loc[ed, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
    # Arbitrarily choose a day for ecodormancy to start, currently november 30th
    for ed in np.argwhere(df["YEAR_JDAY"]==334):
        df.loc[ed, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
    
    # Covert phenology to int values
    for i in range(len(df["PHENOLOGY"])):
        if isinstance(df.loc[i,"PHENOLOGY"], str):
            df.loc[i,"PHENOLOGY"] = PHENOLOGY_INT[df["PHENOLOGY"].iloc[i]]'''
   
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

    df_list = []
    stages_list = []

    ys = np.argwhere(df["YEAR_JDAY"] == 1).flatten()

    for i in range(len(ys)):
        # Get the slices of each individual year
        if i == len(ys) - 1:
            year_df = df[ys[i]:].copy().reset_index(drop=True)
        else:
            year_df = df[ys[i]:ys[i+1]].copy().reset_index(drop=True)

            # Get the switch to the dormancy season so we can add endodorm
        for ds in np.argwhere(np.diff(df["DORMANT_SEASON"],prepend=[0]) == 1):
            df.loc[ds, "PHENOLOGY"] = PHENOLOGY_INT["Endodorm"]

        # TODO: With real values
        # Check if any real values are present, if not fill with fake ones
        eco = np.argwhere(year_df["PHENOLOGY"]=="Ecodorm").flatten()
        if len(eco) == 0:
            year_df.loc[0, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
            if len(year_df) >= 334:
                year_df.loc[334, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
        elif len(eco) == 1:
            if eco < 150 and len(year_df) >= 334:
                year_df.loc[334, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
            else: 
                year_df.loc[0, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
        
        endo = np.argwhere(year_df["PHENOLOGY"] == "Endodorm").flatten()
        if len(endo) == 0:
            if year_df.loc[0, "DORMANT_SEASON"] == 0:
                dorm = np.argwhere(np.diff(year_df["DORMANT_SEASON"],prepend=[0]) == 1).flatten()
            else: 
                dorm = np.argwhere(np.diff(year_df["DORMANT_SEASON"],prepend=[1]) == 1).flatten()
            year_df.loc[dorm, "PHENOLOGY"] = PHENOLOGY_INT["Endodorm"]

        # Covert phenology to int values
        for i in range(len(year_df["PHENOLOGY"])):
            if isinstance(year_df.loc[i,"PHENOLOGY"], str):
                year_df.loc[i,"PHENOLOGY"] = PHENOLOGY_INT[year_df["PHENOLOGY"].iloc[i]]
        # Change Phenology dtype
        year_df["PHENOLOGY"] = year_df["PHENOLOGY"].astype('float64')

        # Handle case where a state occurs out of order by removing it
        pheno_changes = np.argwhere(~np.isnan(year_df["PHENOLOGY"] )).flatten()
        pheno_change_vals = year_df.loc[pheno_changes, "PHENOLOGY"].to_numpy().astype('int64')
        for j in range(len(pheno_change_vals)-1):
            if pheno_change_vals[j] > pheno_change_vals[j+1] \
                and pheno_change_vals[j] != PHENOLOGY_INT["Endodorm"]:
                year_df.loc[pheno_changes[j], "PHENOLOGY"] = np.nan

        # Forward fill with non-na values
        year_df["PHENOLOGY"] = year_df["PHENOLOGY"].ffill().astype('int64')

        pheno_states = np.unique(year_df["PHENOLOGY"])

        # If there are any nan values in the weather throw out the entire year
        if year_df.isnull().any().any():
            continue
            
        year_stages = []
        if PHENOLOGY_INT["Ecodorm"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Ecodorm"])
        if PHENOLOGY_INT["Endodorm"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Endodorm"])    
        if PHENOLOGY_INT["Budburst/Budbreak"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Budburst/Budbreak"])
        if PHENOLOGY_INT["Full Bloom"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Full Bloom"])
        if PHENOLOGY_INT["Veraison 50%"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Veraison 50%"])
        if PHENOLOGY_INT["Harvest"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Harvest"])

        # Otherwise append
        df_list.append(year_df)
        stages_list.append(year_stages)

    for yr_df in df_list:
        yr_df.drop(columns=["DORMANT_SEASON", "YEAR_JDAY"],inplace=True)
        yr_df = yr_df[COL_ORDERING]
    
    return df_list, stages_list

def load_and_process_data_nondormant(cultivar: str):
    """
    Load and process AgAid GitHub repository data
    Do not assume dormancy data is available
    """
    
    df = pd.read_csv(f"{DATASET_DIRECTORY}"+f'ColdHardiness_Grape_{cultivar}.csv')

    # Remove all grape stages we are not interested in predicting
    df.loc[~df["PHENOLOGY"].isin(PHENO_STAGES),"PHENOLOGY"] = np.nan

    '''# Get the switch to the dormancy season so we can add endodorm
    for ds in np.argwhere(np.diff(df["DORMANT_SEASON"],prepend=[0]) == 1):
        df.loc[ds, "PHENOLOGY"] = PHENOLOGY_INT["Endodorm"]

    # TODO: With real values
    # For backfilling purposes
    for ed in np.argwhere(df["YEAR_JDAY"]==1):
        df.loc[ed, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
    # Arbitrarily choose a day for ecodormancy to start, currently november 30th
    for ed in np.argwhere(df["YEAR_JDAY"]==334):
        df.loc[ed, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
    
    # Covert phenology to int values
    for i in range(len(df["PHENOLOGY"])):
        if isinstance(df.loc[i,"PHENOLOGY"], str):
            df.loc[i,"PHENOLOGY"] = PHENOLOGY_INT[df["PHENOLOGY"].iloc[i]]'''
   
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

    df_list = []
    stages_list = []

    ys = np.argwhere(df["YEAR_JDAY"] == 1).flatten()

    for i in range(len(ys)):
        # Get the slices of each individual year
        if i == len(ys) - 1:
            year_df = df[ys[i]:].copy().reset_index(drop=True)
        else:
            year_df = df[ys[i]:ys[i+1]].copy().reset_index(drop=True)

            # Get the switch to the dormancy season so we can add endodorm
        for ds in np.argwhere(np.diff(df["DORMANT_SEASON"],prepend=[0]) == 1):
            df.loc[ds, "PHENOLOGY"] = PHENOLOGY_INT["Endodorm"]

        # TODO: With real values
        # Check if any real values are present, if not fill with fake ones
        eco = np.argwhere(year_df["PHENOLOGY"]=="Ecodorm").flatten()
        if len(eco) == 0:
            year_df.loc[0, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
            if len(year_df) >= 334:
                year_df.loc[334, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
        elif len(eco) == 1:
            if eco < 150 and len(year_df) >= 334:
                year_df.loc[334, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
            else: 
                year_df.loc[0, "PHENOLOGY"] = PHENOLOGY_INT["Ecodorm"]
        
        endo = np.argwhere(year_df["PHENOLOGY"] == "Endodorm").flatten()
        if len(endo) == 0:
            if year_df.loc[0, "DORMANT_SEASON"] == 0:
                dorm = np.argwhere(np.diff(year_df["DORMANT_SEASON"],prepend=[0]) == 1).flatten()
            else: 
                dorm = np.argwhere(np.diff(year_df["DORMANT_SEASON"],prepend=[1]) == 1).flatten()
            year_df.loc[dorm, "PHENOLOGY"] = PHENOLOGY_INT["Endodorm"]

        # Covert phenology to int values
        for i in range(len(year_df["PHENOLOGY"])):
            if isinstance(year_df.loc[i,"PHENOLOGY"], str):
                year_df.loc[i,"PHENOLOGY"] = PHENOLOGY_INT[year_df["PHENOLOGY"].iloc[i]]
        # Change Phenology dtype
        year_df["PHENOLOGY"] = year_df["PHENOLOGY"].astype('float64')

        

        # Handle case where a state occurs out of order by removing it
        pheno_changes = np.argwhere(~np.isnan(year_df["PHENOLOGY"] )).flatten()
        pheno_change_vals = year_df.loc[pheno_changes, "PHENOLOGY"].to_numpy().astype('int64')
        for j in range(len(pheno_change_vals)-1):
            if pheno_change_vals[j] > pheno_change_vals[j+1] \
                and pheno_change_vals[j] != PHENOLOGY_INT["Endodorm"]:
                year_df.loc[pheno_changes[j], "PHENOLOGY"] = np.nan

        # Forward fill with non-na values
        year_df["PHENOLOGY"] = year_df["PHENOLOGY"].ffill().astype('int64')

        pheno_states = np.unique(year_df["PHENOLOGY"])

        if PHENOLOGY_INT["Budburst/Budbreak"] not in pheno_states:
            continue

        if PHENOLOGY_INT["Budburst/Budbreak"] in pheno_states and PHENOLOGY_INT["Endodorm"] in pheno_states \
            and PHENOLOGY_INT["Full Bloom"] not in pheno_states:
            continue

        if PHENOLOGY_INT["Full Bloom"] in pheno_states and PHENOLOGY_INT["Endodorm"] in pheno_states \
            and PHENOLOGY_INT["Veraison 50%"] not in pheno_states:
            continue

        # If there are any nan values in the weather throw out the entire year
        if year_df.isnull().any().any():
            continue

        year_stages = []
        if PHENOLOGY_INT["Ecodorm"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Ecodorm"])
        if PHENOLOGY_INT["Endodorm"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Endodorm"])    
        if PHENOLOGY_INT["Budburst/Budbreak"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Budburst/Budbreak"])
        if PHENOLOGY_INT["Full Bloom"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Full Bloom"])
        if PHENOLOGY_INT["Veraison 50%"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Veraison 50%"])
        if PHENOLOGY_INT["Harvest"] in pheno_states:
            year_stages.append(PHENOLOGY_INT["Harvest"])

        # Only go through the onset of dormancy
        if len(dorm) != 0:
            year_df = year_df[:int(dorm[0])]

        # Otherwise append
        df_list.append(year_df)
        stages_list.append(year_stages)

    for yr_df in df_list:
        yr_df.drop(columns=["DORMANT_SEASON", "YEAR_JDAY"],inplace=True)
        yr_df = yr_df[COL_ORDERING]
    
    return np.array(df_list,dtype=object), np.array(stages_list,dtype=object)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cultivar", type=str, default="Aligote", help='Grape cultivar, see directory for possible cultivars')

    args = parser.parse_args()

    load_and_process_data_nondormant(args.cultivar)
if __name__ == "__main__":

    main()