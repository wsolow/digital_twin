"""File for utils functions. Importantly contains:
    - Args: Dataclass for configuring paths for the WOFOST Environment
    - get_gym_args: function for getting the required arguments for the gym 
    environment from the Args dataclass 

Written by: Will Solow, 2024
"""



import gymnasium as gym
import warnings
import numpy as np 
import pandas as pd
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=UserWarning)

def norm(x):
    """
    Take the norm ignoring nans
    """
    return (x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))

def load_data_files(df_names: list[str]) -> list[pd.DataFrame]:
    """
    Load datafiles as dataframe from CSV and return list
    """
    dfs = []
    for dfn in df_names:
        dfs.append(pd.read_csv(dfn, delimiter=',', index_col=0))

    return dfs

def assert_vars(df:pd.DataFrame, vars:list[str]):
    """
    Assert that required variables are present in dataframe
    """
    for var in vars:
        assert var in df, f"{var} not in data" 

def weighted_avg_and_std(arr:list):
    """
    Return the weighted average and standard deviation by padding arrays

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    max_len = max(len(row) for row in arr)

    padded_arr = []
    weights = []
    for row in arr:
        if len(row) < max_len:
            filled_row = np.pad(row, (0, max_len - len(row)), mode='constant', constant_values=0)
            padded_arr.append(filled_row)
            weights.append(np.pad(np.ones(len(row)), (0, max_len - len(row)), mode='constant', constant_values=0))
        else:
            padded_arr.append(row)
            weights.append(np.ones(len(row)))

    average = np.average(padded_arr, weights=weights,axis=0)
    # Fast and numerically precise:
    variance = np.average((padded_arr-average)**2, weights=weights,axis=0)
    return (average, np.sqrt(variance))