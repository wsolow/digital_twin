import warnings
import numpy as np
import argparse
from omegaconf import OmegaConf
from bayesian_optimizer import BayesianNonDormantOptimizer
import os, yaml
import pickle as pkl
import random
import torch

def find_pickle_files(root_dir, extensions=('.pkl', '.pickle')):
    pickle_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(extensions):
                full_path = os.path.join(dirpath, filename)
                pickle_files.append(full_path)

    return sorted(pickle_files, key=lambda x: os.path.basename(x).lower())

def main():
    warnings.filterwarnings("ignore",category=UserWarning)
    np.set_printoptions(suppress=True, precision=3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default", type=str, help="Path to Config")
    parser.add_argument("--cultivar", default="Chardonnay",type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    config = OmegaConf.load(f"configs/{args.config}.yaml")
    config.cultivar = args.cultivar

    pkl_files = find_pickle_files(f"logs/calib/{config.cultivar}")
    
    rmse_avg_test = np.zeros(3)
    rmse_avg_train = np.zeros(3)

    num_runs = 5
    for seed in range(num_runs):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        with open(f"{pkl_files[seed]}", "rb") as f:
            params_1 = pkl.load(f)
    
        optim = BayesianNonDormantOptimizer(config)
        rmse_avg_train += optim.plot_comparison_bar(params_1, path=f"logs/comparisons/{config.cultivar}/", train=True)
        rmse_avg_test += optim.plot_comparison_bar(params_1, path=f"logs/comparisons/{config.cultivar}/", train=False)

    with open("data_train.txt", "a") as f:
        f.write(f"{config.cultivar},")
        data = np.round(np.concatenate((rmse_avg_train/num_runs, [0], rmse_avg_test/num_runs, [0])), decimals=2)
        data_str = ",".join(map(str, data))
        f.write(data_str + "\n")  # add newline at the end
        #np.savetxt(f, np.concatenate((rmse_avg_train/num_runs, [0], rmse_avg_test/num_runs, [0])).flatten(), delimiter=',', fmt='%s')


if __name__ == "__main__":
    main()