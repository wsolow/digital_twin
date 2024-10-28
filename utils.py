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

import grape_gym.wrappers.wrappers as wrappers
from grape_gym.args import NPK_Args

warnings.filterwarnings("ignore", category=UserWarning)

@dataclass
class Args:
    """Dataclass for configuration a Gym environment
    """

    """Parameters for the NPK Gym Environment"""
    npk_args: NPK_Args

    """Environment ID"""
    env_id: str = "grape-lnpkw-v0"
    """Env Reward Function"""
    env_reward: str = "default"

    """Location of data folder which contains multiple runs"""
    save_folder: str = "data/"

    """Path"""
    base_fpath: str = "/Users/wsolow/Projects/digital_twin/"
    """Relative path to agromanagement configuration file"""
    agro_fpath: str = "env_config/agro_config/grape_agro.yaml"
    """Relative path to crop configuration folder"""
    crop_fpath: str = "env_config/crop_config/"
    """Relative path to site configuration foloder"""
    site_fpath: str = "env_config/site_config/"
    """Relative path to the state units """
    unit_fpath: str = "env_config/param_units.yaml"
    """Relative path to the state names"""
    name_fpath: str = "env_config/param_names.yaml"

    """Policy name if using a policy in the policies.py file"""
    policy_name: str = None

    """Year range, incremented by 1"""
    year_range: list = field(default_factory = lambda: [1984, 2000])
    """Latitude Range, incremented by .5"""
    lat_range: list = field(default_factory = lambda: [50, 50])
    """Longitude Range of values, incremented by .5"""
    long_range: list = field(default_factory = lambda: [5, 5])

    """Agent type, for generating data"""
    agent_type: str = None
    """Agent path, for loading .pt agents"""
    agent_path: str = None

def get_gym_args(args: Args):
    """
    Returns the Environment ID and required arguments for the WOFOST Gym
    Environment

    Arguments:
        Args: Args dataclass
    """
    env_kwargs = {'args': args.npk_args, 'base_fpath': args.base_fpath, \
                  'agro_fpath': args.agro_fpath,'site_fpath': args.site_fpath, 
                  'crop_fpath': args.crop_fpath }
    
    return args.env_id, env_kwargs

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

def wrap_env_reward(env: gym.Env, args):
    """
    Function to wrap the environment with a given reward function
    Based on the reward functions created in the wofost_gym/wrappers/
    """

    if args.env_reward == "RewardFertilizationCostWrapper":
        print('Fertilization Cost Reward Function')
        return wrappers.RewardFertilizationCostWrapper(env)
    elif args.env_reward == 'RewardFertilizationThresholdWrapper':
        print('Fertilization Threshold Reward Function')
        return wrappers.RewardFertilizationThresholdWrapper(env, max_n=args.max_n, max_p = args.max_p, max_k=args.max_k, max_w=args.max_w)
    else:
        return env