"""
Class for optimizing the phenology by iteratively adding more data

"""

import argparse
import numpy as np
import load_data as ld
from omegaconf import OmegaConf
from bayesian_optimizer import BayesianNonDormantOptimizer
import warnings
from datetime import datetime
import os
import copy
from digtwin import digital_twin as dt
import matplotlib.pyplot as plt

PHENOLOGY_INT = {"Ecodorm":0, "Budbreak":1, "Flowering":2, "Veraison":3, "Ripe":4, "Endodorm":5}

class IterativeOptimizer():

    def __init__(self, config):
        self.config_file = f"{os.getcwd()}/{config.model_config_fpath}"
        self.digtwin = dt.DigitalTwin(config_fpath=self.config_file)
        self.params = self.digtwin.get_param_dict()

        self.loss_func = BayesianNonDormantOptimizer.compute_SUM
        
        self.config = config
        self.cultivar = config.cultivar
        self.losses = []

        if self.cultivar is not None:
            self.data_list, self.stage_list = ld.load_and_process_data_nondormant(config.cultivar)
        else:
            raise Exception("Cultivar is `None`")
        
    def optimize(self):
        """
        Optimize the years sequentially by adding data
        """
        self.fpath = f"logs/iterative/{self.cultivar}/{self.cultivar}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        os.makedirs(self.fpath, exist_ok=True)

        args = np.arange(len(self.data_list))
        np.random.shuffle(args)

        self.data_list = self.data_list[args]
        self.stage_list = self.stage_list[args]
        opt_losses = np.zeros(len(self.data_list))

        for i in range(len(self.data_list)):
            optim = BayesianNonDormantOptimizer(config=self.config, 
                    data_list=self.data_list[:i+1], stage_list=self.stage_list[:i+1])
            optim.optimize(path=f"{self.fpath}/{i}")

            self.params = copy.deepcopy(optim.opt_params)
            optim.save_model(path=f"{self.fpath}/{i}/model.pkl")

            optim.plot()
            for j in range(optim.n_stages):
                optim.plot_gp(j)
            
            opt_losses[i] = self.compute_loss()


        self.losses.append(opt_losses)

    def warm_start_optimize(self):
        """
        Optimize the years sequentially by adding data and warm starting the GP
        """
        self.fpath = f"logs/iterative_warm_start/{self.cultivar}/{self.cultivar}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        os.makedirs(self.fpath, exist_ok=True)

        args = np.arange(len(self.data_list))
        np.random.shuffle(args)

        self.data_list = self.data_list[args]
        self.stage_list = self.stage_list[args]
        opt_losses = np.zeros(len(self.data_list))

        gp = None
        for i in range(len(self.data_list)):
            optim = BayesianNonDormantOptimizer(config=self.config, 
                    data_list=self.data_list[:i+1], stage_list=self.stage_list[:i+1])
            optim.optimize(path=f"{self.fpath}/{i}", gp=gp)

            gp = [copy.deepcopy(optim.gps[0][-1]), copy.deepcopy(optim.gps[1][-1]),\
                  copy.deepcopy(optim.gps[2][-1]),copy.deepcopy(optim.gps[3][-1])]

            self.params = copy.deepcopy(optim.opt_params)
            optim.save_model(path=f"{self.fpath}/{i}/model.pkl")

            optim.plot()
            for j in range(optim.n_stages):
                optim.plot_gp(j)
            
            opt_losses[i] = self.compute_loss()

        self.losses.append(opt_losses)

    def optimize_with_val(self):
        """
        Optimize the years sequentially by adding data with a validation set
        """
        self.fpath = f"logs/iterative_val/{self.cultivar}/{self.cultivar}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        os.makedirs(self.fpath, exist_ok=True)

        args = np.arange(len(self.data_list))
        np.random.shuffle(args)
        test = 2 * (len(self.data_list) // 3)

        data_train = self.data_list[args][:test]
        stage_train = self.stage_list[args][:test]
        data_test = self.data_list[args][-test:]
        stage_test = self.stage_list[args][-test:]
        opt_losses = np.zeros(len(data_train))

        for i in range(len(data_train)):
            optim = BayesianNonDormantOptimizer(config=self.config, 
                    data_list=data_train[:i+1], stage_list=stage_train[:i+1])
            optim.optimize(path=f"{self.fpath}/{i}")

            self.params = copy.deepcopy(optim.opt_params)
            optim.save_model(path=f"{self.fpath}/{i}/model.pkl")

            optim.plot()
            for j in range(optim.n_stages):
                optim.plot_gp(j)
            
            opt_losses[i] = self.compute_loss(data_test)


        self.losses.append(opt_losses)

    def warm_start_optimize_with_val(self):
        """
        Optimize the years sequentially by adding data and warm starting the GP
        """
        self.fpath = f"logs/iterative_warm_start_val/{self.cultivar}/{self.cultivar}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        os.makedirs(self.fpath, exist_ok=True)

        args = np.arange(len(self.data_list))
        np.random.shuffle(args)
        test = 2 * (len(self.data_list) // 3)

        data_train = self.data_list[args][:test]
        stage_train = self.stage_list[args][:test]
        data_test = self.data_list[args][-test:]
        stage_test = self.stage_list[args][-test:]
        opt_losses = np.zeros(len(data_train))

        gp = None
        for i in range(len(data_train)):
            optim = BayesianNonDormantOptimizer(config=self.config, 
                    data_list=data_train[:i+1], stage_list=stage_train[:i+1])
            optim.optimize(path=f"{self.fpath}/{i}", gp=gp)

            gp = [copy.deepcopy(optim.gps[0][-1]), copy.deepcopy(optim.gps[1][-1]),\
                  copy.deepcopy(optim.gps[2][-1]),copy.deepcopy(optim.gps[3][-1])]

            self.params = copy.deepcopy(optim.opt_params)
            optim.save_model(path=f"{self.fpath}/{i}/model.pkl")

            optim.plot()
            for j in range(optim.n_stages):
                optim.plot_gp(j)
            
            opt_losses[i] = self.compute_loss(data_test)

        self.losses.append(opt_losses)

    def compute_loss(self, data_list:list=None):
        """
        Compute the loss of the phenology model
        """
        loss = 0 
        if data_list is None:
            data_list = self.data_list

        for i in range(len(data_list)):
            true_output, model_output = self.digtwin.run_from_data(data_list[i], args=self.params, run_till=True)
            loss += self.loss_func(true_output, model_output, None, None)
        return -loss
    
    def plot_losses(self):
        """
        Plot the losses accumulated by the optimizer
        """
        loss_mean = np.mean(self.losses, axis=0)
        loss_std = np.std(self.losses, axis=0)

        x = np.arange(len(self.losses))
        fig, ax = plt.subplots()
        ax.plot(x, loss_mean, marker=".")
        ax.fill_between(x, loss_mean-loss_std, loss_mean+loss_std, alpha=.5)
        ax.set_xlabel("Number of Years of Data")
        ax.set_xticks(x)
        ax.set_ylabel("Average Loss")
        plt.savefig(f"{self.fpath}/losses.png")

        plt.close()


def main():
    warnings.filterwarnings("ignore",category=UserWarning)
    np.set_printoptions(suppress=True, precision=3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default", type=str, help="Path to Config")
    args = parser.parse_args()

    config = OmegaConf.load(f"configs/{args.config}.yaml")

    optim = IterativeOptimizer(config)
    for _ in range(config.num_runs):
        optim.optimize()
    optim.plot_losses()

    
if __name__ == "__main__":
    main()