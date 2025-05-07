"""File for utils functions. Importantly contains:
    - Args: Dataclass for configuring paths for the WOFOST Environment
    - get_gym_args: function for getting the required arguments for the gym 
    environment from the Args dataclass 

Written by: Will Solow, 2024
"""
import pickle, argparse, warnings, os, sys
import numpy as np
from bayesian_optimizer import BayesianNonDormantOptimizer
from omegaconf import OmegaConf
import yaml

def find_config_yaml_dirs(start_dir=".", cultivar="Aligote"):
    config_dirs = []
    print(cultivar)
    for root, dirs, files in os.walk(start_dir):
        if f"0_{cultivar}.pkl" in files:
            relative_path = os.path.relpath(root, start_dir)
            config_dirs.append(relative_path)
    return config_dirs

def main():
    warnings.filterwarnings("ignore",category=UserWarning)
    np.set_printoptions(suppress=True, precision=3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default", type=str, help="Path to Config")
    parser.add_argument("--cultivar", default="Aligote",type=str)
    args = parser.parse_args()
    fpath = f'logs/2024_data_calib/'
    full_path = find_config_yaml_dirs(fpath, cultivar=args.cultivar)

    with open(f"{fpath}/{full_path[0]}/0_{args.cultivar}.pkl", "rb") as f:
        params = pickle.load(f)
    
    params = {k: float(np.round(v, decimals=2)) for k,v in params.items()}

    with open(f"models/new_grape.yaml", "a") as g:
        yaml.dump({args.cultivar: params}, g)

if __name__ == "__main__":
    main()

