"""
Class for optimizing the phenology
"""

import argparse
import bayes_opt
from omegaconf import OmegaConf
from bayes_opt import acquisition
from bayes_opt import BayesianOptimization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from digtwin import digital_twin as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
import yaml
import load_data as ld
import sys
import os
from datetime import datetime
import copy
from matplotlib import cm
from collections import deque
import warnings

PHENOLOGY_INT = {"Ecodorm":0, "Budbreak":1, "Flowering":2, "Veraison":3, "Ripe":4, "Endodorm":5}

class BayesianNonDormantOptimizer():
    """
    Optimize each grape phenology stage iteratively 
    """
    def __init__(self, config):
        
        self.stages = ["Budbreak", "Flowering", "Veraison", "Ripe"]
        self.n_stages = 4
        self.config_file = config.model_config_fpath
        self.init_points = config.init_points
        self.n_iter = config.n_iter
        self.config = config

        if config.acq == "UCB":
            self.acq = acquisition.UpperConfidenceBound()
        elif config.acq == "EI":
            self.acq = acquisition.ExpectedImprovement(xi=.01)
        else:
            raise Exception(f"Unexpected Acquisition Function {config.acq}")
        
        data, cult = self.load_config_data()

        self.cultivar = config.cultivar
        if self.cultivar is not None:
            self.data_list, self.stage_list = ld.load_and_process_data_nondormant(config.cultivar)
        else:
            self.data_list = [data]
            self.stage_list = list(PHENOLOGY_INT.values())

        self.digtwin = dt.DigitalTwin(config_fpath=self.config_file)
        
        self.params = self.digtwin.get_param_dict()
        self.all_params = []
        self.opt_params = copy.deepcopy(self.params)

        self.pbounds = [{"TBASEM": (0,15),"TSUMEM":(10, 500)}, # Bud break
                        {"TEFFMX":(20,45), "TSUM1":(100, 1000)}, # Flowering
                        {"TEFFMX":(20,45), "TSUM2":(100,1000)}, # Veraison 
                        {"TEFFMX":(20,45), "TSUM3":(100,1000)}] # Ripe
        
        self.samples = [[] for _ in range(self.n_stages)]
        self.optimizers = [None]*self.n_stages
        self.gps = [[] for _ in range(self.n_stages)]
   
    def optimize(self):
        """
        Iteratively optimize the crop model by stage
        """
        self.stage_params = []
        self.fpath = f"logs/{self.cultivar}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        os.makedirs(self.fpath, exist_ok=True)
        with open(f"{self.fpath}/config.yaml", "w") as fp:
            OmegaConf.save(config=self.config, f=fp.name)

        for i in range(self.n_stages):
            self.params = copy.deepcopy(self.opt_params)
            optimizer = BayesianOptimization(f=None, pbounds=self.pbounds[i], \
                                             acquisition_function=self.acq, allow_duplicate_points=True)
            
            optimizer._gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=1e-10,
            normalize_y=False,
            n_restarts_optimizer=5,
            random_state=0,
            )

            samples = self.maximize(optimizer, \
                                         list(self.pbounds[i].keys()), self.stages[i], i)
            self.samples[i] = np.array(samples)
            self.optimizers[i] = optimizer

    def maximize(self, optimizer, vars:list, stage:str, ind:int):
        """
        Function to iteratively maximize the gaussian process
        """
        init_points = deque()
        for _ in range(self.init_points):
            init_points.append(optimizer._space.random_sample())

        iteration = 0
        samples = []
        max_val = -np.inf
        while init_points or iteration < self.n_iter:
            try:
                x_probe = init_points.popleft()
            except IndexError:
                x_probe = optimizer.suggest()
                iteration += 1

            # Update parameters
            if not isinstance(x_probe,dict):
                x_probe = dict(zip(vars,x_probe))

            self.update_params(x_probe)
            self.all_params.append(self.params)
            
            # Run the model and compute loss
            target = self.compute_loss(stage)
            optimizer.register(target=target, params=x_probe)
            samples.append([target, *x_probe.values()])

            if target > max_val:
                self.opt_params = copy.deepcopy(self.params)
                max_val = target
            
            self.gps[ind].append(copy.deepcopy(optimizer._gp))
            
        return samples

    def compute_loss(self, stage):
        """
        Compute the loss of the phenology model
        """
        loss = 0 
        for i in range(len(self.data_list)):
            if stage == "Endodorm" or stage == "Ecodorm":
                pass
            else:
                # For all stages, check if the previous stage is there
                if PHENOLOGY_INT[stage]-1 not in self.stage_list[i]:
                    continue
                
                # If not ripe, check the current stage is there
                if stage != "Ripe":
                    if PHENOLOGY_INT[stage] not in self.stage_list[i]:
                        continue
                
                # For bud break and flowering, check that the next stage is there
                if stage == "Budbreak" or stage == "Flowering":
                    if PHENOLOGY_INT["Endodorm"] in self.stage_list[i] and \
                        PHENOLOGY_INT[stage]+1 not in self.stage_list[i]:
                        continue
                
            true_output, model_output = self.digtwin.run_from_data(self.data_list[i], args=self.params)
            loss += self.compute_SUM_SLICE(true_output, model_output, stage, self.stage_list[i])

        return loss

    def update_params(self, params):
        """
        Update the parameters of the model
        """
        # Phenology Parameters
        if "TBASEM" in params.keys():
            self.params["TBASEM"] = params["TBASEM"]
        if "TEFFMX" in params.keys():
            self.params["TEFFMX"] = params["TEFFMX"]
        if "TSUMEM" in params.keys():
            self.params["TSUMEM"] = params["TSUMEM"]
        if "TSUM1" in params.keys():
            self.params["TSUM1"] = params["TSUM1"]
        if "TSUM2" in params.keys():
            self.params["TSUM2"] = params["TSUM2"]
        if "TSUM3" in params.keys():
            self.params["TSUM3"] = params["TSUM3"]
        if "TSUM4" in params.keys():
            self.params["TSUM4"] = params["TSUM4"]
        if "MLDORM" in params.keys():
            self.params["MLDORM"] = params["MLDORM"]
        if "Q10C" in params.keys():
            self.params["Q10C"] = params["Q10C"]
        if "CSUMDB" in params.keys():
            self.params["CSUMDB"] = params["CSUMDB"]
        
    def compute_SUM_SLICE(self, true, model, stage, val_stages):
        """
        Compute the error in days between the true data and current model
        only for the current and previous stage
        """

        curr_stage = (PHENOLOGY_INT[stage]) % len(PHENOLOGY_INT)
        prev_stage = (PHENOLOGY_INT[stage]-1) % len(PHENOLOGY_INT)

        model_output = model["PHENOLOGY"].to_numpy()
        true_output = true["PHENOLOGY"].to_numpy()

        true_stage_args = np.argwhere(true_output == curr_stage).flatten()
        model_stage_args = np.argwhere(model_output == curr_stage).flatten()
        true_prev_args = np.argwhere(true_output == prev_stage).flatten()
        model_prev_args = np.argwhere(model_output == prev_stage).flatten()

        args = np.unique(np.concatenate((true_stage_args, model_stage_args, true_prev_args, model_prev_args)))
        
        if len(args) == 0:
            return 0
        true_output = true_output[args]
        model_output = model_output[args]

        return -np.sum(true_output != model_output)
    
    def load_config_data(self):
        config = yaml.safe_load(open(self.config_file))
        twin_config = config["ModelConfig"]
        data = pd.read_csv(os.path.join(twin_config["base_fpath"], twin_config["digtwin_file"]), index_col=0)
        return data, twin_config["targ_cultivar"]

    def save_model(self, path:str):
        """
        Save the current model of the digital twin
        """
        self.digtwin.save_model(path)

    def plot_gp(self, ind):
        """
        Plot the Gaussian Process Mean of all parameters
        """
        os.makedirs(f"{self.fpath}/GP",exist_ok=True)

        gran = 50
        levs = 100
        
        fig, ax = plt.subplots(2,2)
        x, y = self.pbounds[ind].values()
        x = np.linspace(*x, num=gran)
        y = np.linspace(*y, num=gran)
        X, Y = np.meshgrid(x,y)
        inp = np.array(np.meshgrid(x,y)).T.reshape(-1,2)

        #Z_mean, Z_cov = self.gps[i].predict(inp, return_cov=True)
        Z_mean, Z_std = self.optimizers[ind]._gp.predict(inp, return_std=True)
        Z_acq = -1 * self.optimizers[ind]._acquisition_function.base_acq(Z_mean, Z_std)
        
        Z_mean = Z_mean.reshape(gran,gran).T
        Z_std = Z_std.reshape(gran,gran).T
        Z_acq = Z_acq.reshape(gran,gran).T
        max = np.argmax(self.samples[ind][:,0]).flatten()[0]

        # GP Mean
        im1 = ax[0,0].contourf(X, Y, Z_mean, cmap=cm.jet,levels=levs)
        ax[0,0].scatter(self.samples[ind][:,1], self.samples[ind][:,2], marker="x",c='k')
        ax[0,0].scatter(self.samples[ind][max,1], self.samples[ind][max,2], marker="x",c='deeppink')
        fig.colorbar(im1, ax=ax[0,0])
        ax[0,0].set_title("Gaussian Process Mean")

        # GP Variance
        im2 = ax[1,0].contourf(X,Y,Z_std, cmap=cm.jet,levels=levs)
        ax[1,0].scatter(self.samples[ind][:,1], self.samples[ind][:,2], marker="x",c='k')
        ax[1,0].scatter(self.samples[ind][max,1], self.samples[ind][max,2], marker="x",c='deeppink')
        fig.colorbar(im2, ax=ax[1,0])
        ax[1,0].set_title("Gaussian Process Variance")

        # Acq Function
        im2 = ax[1,1].contourf(X,Y,Z_acq, cmap=cm.jet,levels=levs)
        ax[1,1].scatter(self.samples[ind][:,1], self.samples[ind][:,2], marker="x",c='k')
        ax[1,1].scatter(self.samples[ind][max,1], self.samples[ind][max,2], marker="x",c='deeppink')
        fig.colorbar(im2, ax=ax[1,1])
        ax[1,1].set_title("Acquisition Function")

        plt.savefig(f"{self.fpath}/GP/{self.cultivar}_{self.stages[ind]}_Final_GP.png")

    def plot(self):
        os.makedirs(f"{self.fpath}/Phenology",exist_ok=True)
        for data in self.data_list:
            true_output, model_output = self.digtwin.run_from_data(data, args=self.opt_params)
            x=np.arange(len(true_output))
            plt.figure()
            plt.plot(x, true_output["PHENOLOGY"],label='True Data')
            plt.plot(x, model_output["PHENOLOGY"], label='Calibrated Model')
            start = true_output["DATE"].iloc[0]
            end = true_output["DATE"].iloc[-1]
            plt.title(f"{self.cultivar} Phenology from {start} to {end}")
            plt.ylabel('Phenology Stage')
            plt.yticks(ticks=[0,1,2,3,4,5], labels=['Ecodorm', 'Bud Break', 'Flower', 'veraison', 'Ripe', 'Endodorm'], rotation=45)
            plt.xlabel(f'Days since {start}')
            plt.legend()
            plt.savefig(f"{self.fpath}/Phenology/{self.cultivar}_PHENOLOGY_{start}.png")

    def animate_gp(self, ind):
        gran = 50
        levs = 100
        cmap = cm.jet

        os.makedirs(f"{self.fpath}/GP",exist_ok=True)

        x, y = self.pbounds[ind].values()
        x = np.linspace(*x, num=gran)
        y = np.linspace(*y, num=gran)
        X, Y = np.meshgrid(x,y)
        inp = np.array(np.meshgrid(x,y)).T.reshape(-1,2)

        Z_mean, Z_std = self.optimizers[ind]._gp.predict(inp, return_std=True)
        Z_acq = -1 * self.optimizers[ind]._acquisition_function.base_acq(Z_mean, Z_std)
        mean_min = np.inf
        mean_max = -np.inf
        std_min = np.inf
        std_max = -np.inf
        acq_min = np.inf
        acq_max = -np.inf

        # Find normalizing range for colorbars
        for j in range(len(self.gps[ind])):
            Z_mean, Z_std = self.gps[ind][j].predict(inp, return_std=True)
            Z_acq = -1 * self.optimizers[ind]._acquisition_function.base_acq(Z_mean, Z_std)

            mean_min = np.minimum(mean_min, np.min(Z_mean))
            mean_max = np.maximum(mean_max, np.max(Z_mean))

            std_min = np.minimum(std_min, np.min(Z_std))
            std_max = np.maximum(std_max, np.max(Z_std))

            acq_min = np.minimum(acq_min, np.min(Z_acq))
            acq_max = np.maximum(acq_max, np.max(Z_acq))

        norm_mean = Normalize(vmin=mean_min, vmax=mean_max)
        norm_std = Normalize(vmin=std_min, vmax=std_max)
        norm_acq = Normalize(vmin=acq_min, vmax=acq_max)

        fig, ax = plt.subplots(2,2)
        fig.colorbar(cm.ScalarMappable(norm=norm_mean, cmap=cmap), ax=ax[0,0])
        fig.colorbar(cm.ScalarMappable(norm=norm_std, cmap=cmap), ax=ax[1,0])
        fig.colorbar(cm.ScalarMappable(norm=norm_acq, cmap=cmap), ax=ax[1,1])

        def update(i):
            for a in ax.flatten():
                a.cla()
                for collection in a.collections:
                    collection.remove()
            Z_mean, Z_std = self.gps[ind][i].predict(inp, return_std=True)
            Z_acq = -1 * self.optimizers[ind]._acquisition_function.base_acq(Z_mean, Z_std)
            
            Z_mean = Z_mean.reshape(gran,gran).T
            Z_std = Z_std.reshape(gran,gran).T
            Z_acq = Z_acq.reshape(gran,gran).T

            max = np.argmax(self.samples[ind][:i+1,0])

            # GP Mean
            im1 = ax[0,0].contourf(X, Y, Z_mean, cmap=cmap,levels=levs, norm=norm_mean)
            ax[0,0].scatter(self.samples[ind][:i+1,1], self.samples[ind][:i+1,2], marker="x",c='k')
            ax[0,0].scatter(self.samples[ind][max,1], self.samples[ind][max,2], marker="x",c='deeppink')

            # GP Variance
            im2 = ax[1,0].contourf(X,Y,Z_std, cmap=cmap,levels=levs, norm=norm_std)
            ax[1,0].scatter(self.samples[ind][:i+1,1], self.samples[ind][:i+1,2], marker="x",c='k')
            ax[1,0].scatter(self.samples[ind][max,1], self.samples[ind][max,2], marker="x",c='deeppink')
            
            # Acq Function
            im2 = ax[1,1].contourf(X,Y,Z_acq, cmap=cmap,levels=levs, norm=norm_acq)
            ax[1,1].scatter(self.samples[ind][:i+1,1], self.samples[ind][:i+1,2], marker="x",c='k')
            ax[1,1].scatter(self.samples[ind][max,1], self.samples[ind][max,2], marker="x",c='deeppink')
            ax[1,1].axvline(self.samples[ind][i,1], c='k')
            ax[1,1].axhline(self.samples[ind][i,2], c='k')

        frames = np.linspace(0, 10, 100)
        ani = animation.FuncAnimation(fig, update, frames=len(self.gps[ind]), interval=500)
        writer = animation.FFMpegWriter(
           fps=2, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(f"{self.fpath}/GP/{self.cultivar}_{self.stages[ind]}_GP_Animiation.mp4", writer=writer)
     
def main():
    warnings.filterwarnings("ignore",category=UserWarning)
    np.set_printoptions(suppress=True, precision=3)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default", type=str, help="Path to Config")
    args = parser.parse_args()

    config = OmegaConf.load(f"configs/{args.config}.yaml")

    optim = BayesianNonDormantOptimizer(config)
    optim.optimize()
    for i in range(optim.n_stages):
        optim.plot_gp(i)
        optim.animate_gp(i)
    
    optim.plot()

    
if __name__ == "__main__":
    main()