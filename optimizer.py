from bayes_opt import BayesianOptimization
from digtwin import digital_twin as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import load_data as ld
import sys
import os
import datetime
import copy

PHENOLOGY_INT = {"Ecodorm":0, "Budbreak":1, "Flowering":2, "Verasion":3, "Ripe":4, "Endodorm":5}

class BayesianDormancyOptimizer():
    """
    Optimize each grape phenology stage iteratively 
    """
    def __init__(self, model_config_fpath:str=None, cultivar=None):
        
        self.config_file = model_config_fpath
        self.init_points = 10
        self.n_iter = 100

        data, cult = self.load_config_data()

        self.cultivar = cultivar
        if cultivar is not None:
            self.data_list, self.stage_list = ld.load_and_process_data_dormant(cultivar)
        else:
            self.data_list = [data]
            self.stage_list = list(PHENOLOGY_INT.values())

        self.digtwin = dt.DigitalTwin(config_fpath=self.config_file)
        
        self.params = self.digtwin.get_param_dict()

        self.set_pbounds()

    def optimize(self):

        optimizer = BayesianOptimization(f=self.crop_optimizer, pbounds=self.pbounds)

        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

        self.opt_params = optimizer.max['params']

        self.all_params = []
        for i, res in enumerate(optimizer.res):
            self.all_params.append(res['params'])

        return self.opt_params, self.all_params
    
    def optimize_stages(self):
        """
        Optimize crop model via individual stages
        """
        stage_opts = [self.eco_stage_optimizer, self.endo_stage_optimizer, self.budbreak_stage_optimizer, \
                            self.flowering_stage_optimizer, self.verasion_stage_optimizer,\
                            self.ripe_stage_optimizer]
        
        pbounds = [self.eco_pbounds, self.endo_pbounds, self.budbreak_pbounds, self.flowering_pbounds, \
                   self.verasion_pbounds, self.ripe_pbounds]
        
        for i in range(len(stage_opts)):
            optimizer = BayesianOptimization(f=stage_opts[i], pbounds=pbounds[i])
            optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

            self.update_params(optimizer.max["params"])

        self.opt_params = self.params
        return self.opt_params

    def eco_stage_optimizer(self, Q10C, CSUMDB):
        """
        Optimizer function for ecodormancy
        """
        params = copy.deepcopy(self.params)
        params["Q10C"] = Q10C
        params["CSUMDB"] = CSUMDB

        return self.optimizer(params, "Ecodorm")
    
    def budbreak_stage_optimizer(self, TBASEM, TSUMEM):
        """
        Optimizer function for budbreak
        """
        params = copy.deepcopy(self.params)
        params["TBASEM"] = TBASEM
        params["TSUMEM"] = TSUMEM

        return self.optimizer(params, "Budbreak")
    
    def flowering_stage_optimizer(self, TSUM1, TEFFMX):
        """
        Optimizer function for flowering
        """
        params = copy.deepcopy(self.params)
        params["TSUM1"] = TSUM1
        params["TEFFMX"] = TEFFMX
    
        return self.optimizer(params, "Flowering")
    
    def verasion_stage_optimizer(self, TSUM2, TEFFMX):
        """
        Optimizer function for verasion
        """
        params = copy.deepcopy(self.params)
        params["TSUM2"] = TSUM2
        params["TEFFMX"] = TEFFMX

        return self.optimizer(params, "Verasion")

    def ripe_stage_optimizer(self, TSUM3, TEFFMX):
        """
        Optimizer function for ripe
        """
        params = copy.deepcopy(self.params)
        params["TSUM3"] = TSUM3
        params["TEFFMX"] = TEFFMX

        return self.optimizer(params, "Ripe")
    
    def endo_stage_optimizer(self, MLDORM, TSUM4):
        """
        Optimizer for endodormancy
        """

        params = copy.deepcopy(self.params)
        params["TSUM3"] = TSUM4
        params["MLDORM"] = MLDORM

        return self.optimizer(params, "Endodorm")

    def optimizer(self, params:dict, stage:str):
        loss = 0
        for i in range(len(self.data_list)):
            # Don't account for losses when we have missing info
            # TODO: Sholud be able to omit this and then do it all in 
            # SUM SLICE
            if stage == "Endodorm" or stage == "Ecodorm":
                pass
            elif stage == "Ripe":
                if PHENOLOGY_INT[stage]-1 not in self.stage_list[i]:
                    continue
            else:
                if PHENOLOGY_INT[stage] not in self.stage_list[i]:
                    continue
                if PHENOLOGY_INT[stage]-1 not in self.stage_list[i]:
                    continue

            true_output, model_output = self.digtwin.run_from_data(self.data_list[i], args=params)
            loss += self.compute_SUM_SLICE(true_output, model_output, stage)
        return loss
    
    def update_params(self, params):
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

    def set_pbounds(self):
        self.eco_pbounds = {"Q10C":(.1, 4),
                             "CSUMDB":(10, 500)}
        self.budbreak_pbounds = {"TBASEM": (0,15),
                            "TSUMEM":(10, 500)}
        self.flowering_pbounds = {"TSUM1":(100, 1000),
                                 "TEFFMX":(15,45)}
        self.verasion_pbounds = {"TSUM2":(100,1000),
                                  "TEFFMX":(15,45)}
        self.ripe_pbounds =  {"TSUM3":(100,1000),
                                  "TEFFMX":(15, 45)}
        self.endo_pbounds = {"TSUM4":(100,1000),
                                  "MLDORM":(10, 14)}
          
    def plot(self):
        digtwin = dt.DigitalTwin(config_fpath=self.config_file, data=self.data_list[0],args=self.opt_params)
        true_output, model_output = digtwin.run_all()

        x=np.arange(len(true_output))
        for v in true_output.columns:
            
            if v == "DATE":
                continue
            plt.figure()
            plt.plot(x, true_output[v],label='True Data')
            plt.plot(x, model_output[v], label='Calibrated Model')
            plt.title(v)
            plt.legend()

        plt.show()

    def plot_all(self):
        plt.figure()

        all_output = []
        for params in self.all_params:
            digtwin = dt.DigitalTwin(config_fpath=self.config_file, data=self.data_list[0],args=params)
            true_output, model_output = digtwin.run_all()
            x=np.arange(len(true_output))
            all_output.append(model_output["DVS"].to_numpy())
            plt.plot(x, model_output["DVS"])

        digtwin = dt.DigitalTwin(config_fpath=self.config_file, data=self.data_list[0],args=self.opt_params)
        true_output, model_output = digtwin.run_all()
        x=np.arange(len(true_output))
        plt.plot(x, model_output["DVS"], c='k', label='Best Fit Params')
        plt.plot(x, true_output["DVS"], c='k', linestyle='dashed', label='True Params')

        mean = np.mean(all_output,axis=0)
        std = np.std(all_output,axis=0)
        #plt.plot(x, mean,c='r', label='Average Params')
        #plt.fill_between(x, mean+std, mean-std,alpha=.5,color='r')
        
        plt.ylabel('Numerical Development Stage')
        plt.xlabel('Days Since Jan 1st')
        plt.title('Development Stage Varying Model Parameters')
        plt.legend()
        plt.show()

    def compute_MSE(self, true, model, stage):
        """
        Compute the mean squared error using only the phenological stage
        """
        stage = PHENOLOGY_INT[stage]

        model_output = model["PHENOLOGY"].to_numpy()
        true_output = true["PHENOLOGY"].to_numpy()
        
        true_output = (true_output - np.min(true_output,axis=0)) / (np.max(true_output,axis=0) - np.min(true_output,axis=0)+1e-10)
        model_output = (model_output - np.min(model_output,axis=0)) / (np.max(model_output,axis=0) - np.min(model_output,axis=0)+1e-10)
        
        return -np.mean((true_output-model_output)**2)
    
    def compute_SSE(self,true, model, stage):
        """
        Compute the sum squared error using only the phenological stage
        """
        stage = PHENOLOGY_INT[stage]

        model_output = model["PHENOLOGY"].to_numpy()
        true_output = true["PHENOLOGY"].to_numpy()

        true_output = (true_output - np.min(true_output,axis=0)) / (np.max(true_output,axis=0) - np.min(true_output,axis=0)+1e-10)
        model_output = (model_output - np.min(model_output,axis=0)) / (np.max(model_output,axis=0) - np.min(model_output,axis=0)+1e-10)
        
        return -np.sum((true_output-model_output)**2)
    
    def compute_SSE_SLICE(self, true, model, stage):
        stage = (PHENOLOGY_INT[stage]) % len(PHENOLOGY_INT)
        prev_stage = (PHENOLOGY_INT[stage]-1) % len(PHENOLOGY_INT)

        model_output = model["PHENOLOGY"].to_numpy()
        true_output = true["PHENOLOGY"].to_numpy()

        true_stage_args = np.argwhere(true_output == stage).flatten()
        model_stage_args = np.argwhere(model_output == stage).flatten()
        true_prev_args = np.argwhere(true_output == prev_stage).flatten()
        model_prev_args = np.argwhere(model_output == prev_stage).flatten()

        args = np.unique(np.concatenate((true_stage_args, model_stage_args, true_prev_args, model_prev_args, true)))
        if len(args) == 0:
            return 0
        true_output = true_output[args]
        model_output = model_output[args]

        true_output = (true_output - np.min(true_output,axis=0)) / (np.max(true_output,axis=0) - np.min(true_output,axis=0)+1e-10)
        model_output = (model_output - np.min(model_output,axis=0)) / (np.max(model_output,axis=0) - np.min(model_output,axis=0)+1e-10)
        
        return -np.sum((true_output-model_output)**2)
    
    def compute_SUM_SLICE(self, true, model, stage):
        curr_stage = (PHENOLOGY_INT[stage]) % len(PHENOLOGY_INT)
        prev_stage = (PHENOLOGY_INT[stage]-1) % len(PHENOLOGY_INT)

        # TODO:
        # Only consider the case where the TRUE OUTPUT has that stage
        # And the previous stage 
        # If has previous stage and current stage, compute matching up to 
        # Current stage
        # If has previous stage, current stage, and next stage
        # TODO think about this tmrw 

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
    
    def compute_SUM(self, true, model, stage):
        stage = PHENOLOGY_INT[stage]

        model_output = model["PHENOLOGY"].to_numpy()
        true_output = true["PHENOLOGY"].to_numpy()

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


class BayesianNonDormantOptimizer():
    """
    Optimize each grape phenology stage iteratively 
    """
    def __init__(self, model_config_fpath:str=None, cultivar=None):
        
        self.config_file = model_config_fpath
        self.init_points = 10
        self.n_iter = 20

        data, cult = self.load_config_data()

        self.cultivar = cultivar
        if cultivar is not None:
            self.data_list, self.stage_list = ld.load_and_process_data_nondormant(cultivar)
        else:
            self.data_list = [data]
            self.stage_list = list(PHENOLOGY_INT.values())

        self.digtwin = dt.DigitalTwin(config_fpath=self.config_file)
        
        self.params = self.digtwin.get_param_dict()

        self.set_pbounds()

    def optimize(self):

        optimizer = BayesianOptimization(f=self.crop_optimizer, pbounds=self.pbounds)

        optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

        self.opt_params = optimizer.max['params']

        self.all_params = []
        for i, res in enumerate(optimizer.res):
            self.all_params.append(res['params'])

        return self.opt_params, self.all_params
    
    def optimize_stages(self):
        """
        Optimize crop model via individual stages
        """
        stage_opts = [self.budbreak_stage_optimizer, \
                            self.flowering_stage_optimizer, self.verasion_stage_optimizer,\
                            self.ripe_stage_optimizer]
        
        pbounds = [self.budbreak_pbounds, self.flowering_pbounds, \
                   self.verasion_pbounds, self.ripe_pbounds]
        
        for i in range(len(stage_opts)):
            optimizer = BayesianOptimization(f=stage_opts[i], pbounds=pbounds[i])
            optimizer.maximize(init_points=self.init_points, n_iter=self.n_iter)

            self.update_params(optimizer.max["params"])

        self.opt_params = self.params
        return self.opt_params
 
    def budbreak_stage_optimizer(self, TBASEM, TSUMEM):
        """
        Optimizer function for budbreak
        """
        params = copy.deepcopy(self.params)
        params["TBASEM"] = TBASEM
        params["TSUMEM"] = TSUMEM

        return self.optimizer(params, "Budbreak")
    
    def flowering_stage_optimizer(self, TSUM1, TEFFMX):
        """
        Optimizer function for flowering
        """
        params = copy.deepcopy(self.params)
        params["TSUM1"] = TSUM1
        params["TEFFMX"] = TEFFMX
    
        return self.optimizer(params, "Flowering")
    
    def verasion_stage_optimizer(self, TSUM2, TEFFMX):
        """
        Optimizer function for verasion
        """
        params = copy.deepcopy(self.params)
        params["TSUM2"] = TSUM2
        params["TEFFMX"] = TEFFMX

        return self.optimizer(params, "Verasion")

    def ripe_stage_optimizer(self, TSUM3, TEFFMX):
        """
        Optimizer function for ripe
        """
        params = copy.deepcopy(self.params)
        params["TSUM3"] = TSUM3
        params["TEFFMX"] = TEFFMX

        return self.optimizer(params, "Ripe")

    def optimizer(self, params:dict, stage:str):
        loss = 0
        for i in range(len(self.data_list)):
            # Don't account for losses when we have missing info
            # TODO: Sholud be able to omit this and then do it all in 
            # SUM SLICE
            if stage == "Endodorm" or stage == "Ecodorm":
                pass
            elif stage == "Ripe":
                if PHENOLOGY_INT[stage]-1 not in self.stage_list[i]:
                    continue
            else:
                if PHENOLOGY_INT[stage] not in self.stage_list[i]:
                    continue
                if PHENOLOGY_INT[stage]-1 not in self.stage_list[i]:
                    continue

            true_output, model_output = self.digtwin.run_from_data(self.data_list[i], args=params)
            loss += self.compute_SUM_SLICE(true_output, model_output, stage)
        return loss
    
    def update_params(self, params):
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

    def set_pbounds(self):
        self.budbreak_pbounds = {"TBASEM": (0,15),
                            "TSUMEM":(10, 500)}
        self.flowering_pbounds = {"TSUM1":(100, 1000),
                                 "TEFFMX":(15,45)}
        self.verasion_pbounds = {"TSUM2":(100,1000),
                                  "TEFFMX":(15,45)}
        self.ripe_pbounds =  {"TSUM3":(100,1000),
                                  "TEFFMX":(15, 45)}
          
    def plot(self):
        digtwin = dt.DigitalTwin(config_fpath=self.config_file, data=self.data_list[0],args=self.opt_params)
        true_output, model_output = digtwin.run_all()

        x=np.arange(len(true_output))
        for v in true_output.columns:
            
            if v == "DATE":
                continue
            plt.figure()
            plt.plot(x, true_output[v],label='True Data')
            plt.plot(x, model_output[v], label='Calibrated Model')
            plt.title(v)
            plt.legend()

        plt.show()

    def plot_all(self):
        plt.figure()

        all_output = []
        for params in self.all_params:
            digtwin = dt.DigitalTwin(config_fpath=self.config_file, data=self.data_list[0],args=params)
            true_output, model_output = digtwin.run_all()
            x=np.arange(len(true_output))
            all_output.append(model_output["DVS"].to_numpy())
            plt.plot(x, model_output["DVS"])

        digtwin = dt.DigitalTwin(config_fpath=self.config_file, data=self.data_list[0],args=self.opt_params)
        true_output, model_output = digtwin.run_all()
        x=np.arange(len(true_output))
        plt.plot(x, model_output["DVS"], c='k', label='Best Fit Params')
        plt.plot(x, true_output["DVS"], c='k', linestyle='dashed', label='True Params')

        mean = np.mean(all_output,axis=0)
        std = np.std(all_output,axis=0)
        #plt.plot(x, mean,c='r', label='Average Params')
        #plt.fill_between(x, mean+std, mean-std,alpha=.5,color='r')
        
        plt.ylabel('Numerical Development Stage')
        plt.xlabel('Days Since Jan 1st')
        plt.title('Development Stage Varying Model Parameters')
        plt.legend()
        plt.show()

    def compute_MSE(self, true, model, stage):
        """
        Compute the mean squared error using only the phenological stage
        """
        stage = PHENOLOGY_INT[stage]

        model_output = model["PHENOLOGY"].to_numpy()
        true_output = true["PHENOLOGY"].to_numpy()
        
        true_output = (true_output - np.min(true_output,axis=0)) / (np.max(true_output,axis=0) - np.min(true_output,axis=0)+1e-10)
        model_output = (model_output - np.min(model_output,axis=0)) / (np.max(model_output,axis=0) - np.min(model_output,axis=0)+1e-10)
        
        return -np.mean((true_output-model_output)**2)
    
    def compute_SSE(self,true, model, stage):
        """
        Compute the sum squared error using only the phenological stage
        """
        stage = PHENOLOGY_INT[stage]

        model_output = model["PHENOLOGY"].to_numpy()
        true_output = true["PHENOLOGY"].to_numpy()

        true_output = (true_output - np.min(true_output,axis=0)) / (np.max(true_output,axis=0) - np.min(true_output,axis=0)+1e-10)
        model_output = (model_output - np.min(model_output,axis=0)) / (np.max(model_output,axis=0) - np.min(model_output,axis=0)+1e-10)
        
        return -np.sum((true_output-model_output)**2)
    
    def compute_SSE_SLICE(self, true, model, stage):
        stage = (PHENOLOGY_INT[stage]) % len(PHENOLOGY_INT)
        prev_stage = (PHENOLOGY_INT[stage]-1) % len(PHENOLOGY_INT)

        model_output = model["PHENOLOGY"].to_numpy()
        true_output = true["PHENOLOGY"].to_numpy()

        true_stage_args = np.argwhere(true_output == stage).flatten()
        model_stage_args = np.argwhere(model_output == stage).flatten()
        true_prev_args = np.argwhere(true_output == prev_stage).flatten()
        model_prev_args = np.argwhere(model_output == prev_stage).flatten()

        args = np.unique(np.concatenate((true_stage_args, model_stage_args, true_prev_args, model_prev_args, true)))
        if len(args) == 0:
            return 0
        true_output = true_output[args]
        model_output = model_output[args]

        true_output = (true_output - np.min(true_output,axis=0)) / (np.max(true_output,axis=0) - np.min(true_output,axis=0)+1e-10)
        model_output = (model_output - np.min(model_output,axis=0)) / (np.max(model_output,axis=0) - np.min(model_output,axis=0)+1e-10)
        
        return -np.sum((true_output-model_output)**2)
    
    def compute_SUM_SLICE(self, true, model, stage):
        curr_stage = (PHENOLOGY_INT[stage]) % len(PHENOLOGY_INT)
        prev_stage = (PHENOLOGY_INT[stage]-1) % len(PHENOLOGY_INT)

        # TODO:
        # Only consider the case where the TRUE OUTPUT has that stage
        # And the previous stage 
        # If has previous stage and current stage, compute matching up to 
        # Current stage
        # If has previous stage, current stage, and next stage
        # TODO think about this tmrw 

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
    
    def compute_SUM(self, true, model, stage):
        stage = PHENOLOGY_INT[stage]

        model_output = model["PHENOLOGY"].to_numpy()
        true_output = true["PHENOLOGY"].to_numpy()

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

def main():

    ''' import argparse
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument('--cultivar', type=str, default="Aligote")
    args = argsparser.parse_args()

    optim = BayesianStageOptimizer(model_config_fpath="/Users/wsolow/Projects/digital_twin/env_config/config.yaml", cultivar=args.cultivar)
    optim.optimize_stages()
    optim.save_model(f'models/{args.cultivar}.pkl')
    sys.exit(0)'''
    for cultivar in ld.GRAPE_CULTIVARS:
        print(f'{cultivar}')
        optim = BayesianNonDormantOptimizer(model_config_fpath="/Users/wsolow/Projects/digital_twin/env_config/config.yaml", cultivar=cultivar)
        optim.optimize_stages()
        optim.save_model(f'models/{cultivar}.pkl')

if __name__ == "__main__":
    main()