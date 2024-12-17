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

def main():
    warnings.filterwarnings("ignore",category=UserWarning)
    np.set_printoptions(suppress=True, precision=3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default", type=str, help="Path to Config")
    parser.add_argument("--cultivar", default="Aligote",type=str)
    args = parser.parse_args()

    config = OmegaConf.load(f"configs/{args.config}.yaml")
    config.cultivar = args.cultivar

    optim = BayesianNonDormantOptimizer(config)

    folder =  f"{os.getcwd()}/logs/single/{config.cultivar}"
    
    subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]

    for fpath in subfolders:

        filename = f"{fpath}/{config.cultivar}.pkl"

        with open(filename, "rb") as f:
            optim.opt_params = pickle.load(f)

        optim.plot(path=f"{fpath}/plots/")

if __name__ == "__main__":
    main()

