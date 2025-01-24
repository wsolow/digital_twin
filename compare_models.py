import warnings
import numpy as np
import argparse
from omegaconf import OmegaConf
from bayesian_optimizer import BayesianNonDormantOptimizer
import os, yaml

def main():
    warnings.filterwarnings("ignore",category=UserWarning)
    np.set_printoptions(suppress=True, precision=3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default", type=str, help="Path to Config")
    parser.add_argument("--cultivar", default="Chardonnay",type=str)
    args = parser.parse_args()

    config = OmegaConf.load(f"configs/{args.config}.yaml")
    config.cultivar = args.cultivar

    optim = BayesianNonDormantOptimizer(config)

    with open(f"configs/grape.yaml", "rb") as f:
        params = yaml.safe_load(f)
    
    params_1 = params["CropParameters"]["Varieties"][f"{config.cultivar}"]
    params_2 = params["CropParameters"]["Varieties"][f"{config.cultivar}_Keller"]

    params_1 = {k:v[0] for k, v in params_1.items()}
    params_2 = {k:v[0] for k, v in params_2.items()}

    optim.plot_comparison_bar(params_1, params_2, path=f"logs/comparisons/{config.cultivar}/")

if __name__ == "__main__":
    main()