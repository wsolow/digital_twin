"""File for testing the installation and setup of the WOFOST Gym Environment
with a few simple plots for output 

Written by: Will Solow, 2024
"""

import digtwin
from digtwin.digital_twin import DigitalTwin
import matplotlib.pyplot as plt
import numpy as np
from grape_twin import engine
import sys
import load_data as ld
import yaml
import pandas as pd
import os
from digtwin import digital_twin as dt
import pickle

class CompareModel():

    def __init__(self, config_fpath:str, cultivar:str):

        self.data_list = ld.load_and_process_data(cultivar)
        self.config_fpath = config_fpath
        self.cultivar = cultivar
        self.model_config = self.load_config_data()

        self.digtwin = dt.DigitalTwin(config_fpath=self.config_fpath)

        path = os.path.join(self.model_config["base_fpath"], f'{self.model_config["model_fpath"]}{cultivar}.pkl')
        with open(path, "rb") as fp:
            self.params = pickle.load(fp)

    def run_all(self):
        """
        Run all files in data list
        """
        path = f'{self.model_config["base_fpath"]}{self.model_config["model_fpath"]}/figs'
        for data in self.data_list:
            true_output, model_output = self.digtwin.run_from_data(data, args=self.params)

            x=np.arange(len(true_output))
            plt.figure()
            plt.plot(x, true_output["PHENOLOGY"],label='True Data')
            plt.plot(x, model_output["PHENOLOGY"], label='Calibrated Model')
            start = true_output["DATE"].iloc[0]
            end = true_output["DATE"].iloc[-1]
            plt.title(f"{self.cultivar} Phenology from {start} to {end}")
            plt.ylabel('Phenology Stage')
            plt.yticks(ticks=[0,1,2,3,4,5], labels=['Ecodorm', 'Bud Break', 'Flower', 'Verasion', 'Ripe', 'Endodorm'], rotation=45)
            plt.xlabel(f'Days since {start}')
            plt.legend()
            os.makedirs(f'{path}/{self.cultivar}',exist_ok=True )
            plt.savefig(f'{path}/{self.cultivar}/{self.cultivar}_{start}_{end}.png')

    def load_config_data(self):
        config = yaml.safe_load(open(self.config_fpath))
        twin_config = config["DigTwinConfig"]
        return twin_config


if __name__ == "__main__":

    config_fpath = "/Users/wsolow/Projects/digital_twin/env_config/config.yaml"
    #cultivar = "Aligote"

    for cultivar in ld.GRAPE_CULTIVARS:
    
        model = CompareModel(config_fpath, cultivar)
        model.run_all()

    sys.exit(0)
    
    crop_fpath = "/Users/wsolow/Projects/digital_twin/env_config/"

    gm = engine.GrapePhenologyEngine(config_fpath=config_fpath, crop_fpath=crop_fpath)
    output = gm.run_all()

    output.to_csv('data/test.csv')

    digtwin = DigitalTwin(config_fpath)
    dynamics, model = digtwin.run_all()

    vars = dynamics.columns

    x=np.arange(len(dynamics))
    for v in dynamics.columns:
        
        if v == "DATE":
            continue
        plt.figure()
        plt.plot(x, dynamics[v])
        plt.plot(x, model[v])
        plt.title(v)

    plt.show()

    print(dynamics)
    print(model)
