"""
Digital twin base file 
Written by: Will Solow, 2024
"""
import pandas as pd
import numpy as np
import os
import grape_model as gp
import yaml
import datetime
from datetime import date

from grape_model.engine import GrapePhenologyEngine
from grape_model.nasapower import FileWeatherDataContainer

class DigitalTwin():
    """
    The base class for a digital twin
    """

    def __init__(self, config_fpath:str=None, config:dict=None, data:pd.DataFrame=None, args:dict=None):
        """
        Initialize the digital twin model
        """
        # Load configuration from file
        self.data = data
        if config_fpath is not None:
            self.data, self.crop_config = self._load_config_path(config_fpath)
        if config is not None:
            self.data, self.crop_config = self._load_config_dict(config)

        # Start and end dates for twin
        self.start_date = datetime.datetime.strptime(self.data["DATE"].iloc[0], '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime(self.data["DATE"].iloc[-1], '%Y-%m-%d')
        self.day = self.start_date

        self.output_vars = self.crop_config["output_vars"]
        self.weather_vars = self.crop_config["weather_vars"]

        # Create model 
        self.true_model = GrapeModelFromFile(self.data, self.output_vars+self.weather_vars)
        self.model = GrapeModel(self.crop_config, drv=self.true_model.get_drv(self.day, self.weather_vars), params=args)

    def run(self):
        """
        Run a day in the simulation
        """
        self.day += datetime.timedelta(days=1)

        true_model_output = self.true_model.run(self.day)

        drv = self.true_model.get_drv(self.day, self.weather_vars)

        model_output = self.model.run(self.day, drv)

        return true_model_output, model_output
    
    def run_all(self):
        """
        Run a simulation through termination
        """

        true_output_arr = [self.true_model.get_output()]
        model_output_arr = [self.model.get_output()]

        while self.day < self.end_date:
            true_output, model_output = self.run()
            true_output_arr.append(true_output)
            model_output_arr.append(model_output)
        return pd.DataFrame(true_output_arr, columns=["DATE"]+self.output_vars+self.weather_vars),\
              pd.DataFrame(model_output_arr, columns=["DATE"]+self.output_vars+self.weather_vars)

    def run_from_data(self, data, args:dict=None, run_till=True):
        """
        Run the digital twin given a data file
        """
        # Start and end dates for twin
        self.start_date = datetime.datetime.strptime(data["DATE"].iloc[0], '%Y-%m-%d')
        self.end_date = datetime.datetime.strptime(data["DATE"].iloc[-1], '%Y-%m-%d')
        endo = np.argwhere(data["PHENOLOGY"]==5).flatten()
        self.data = data

        #TODO: Run only until the onset of endodormancy
        if len(endo) > 0:
            self.end_date = datetime.datetime.strptime(data.loc[int(endo[0]),"DATE"], '%Y-%m-%d')
            self.data = data.loc[:endo[0]]
        
        self.day = self.start_date

        self.crop_config["start_date"] = self.start_date
        self.crop_config["end_date"] = self.end_date

        # Update the output variables and weather variables
        all_weather_vars = ["IRRAD", "TMIN", "TMAX", "VAP", "TEMP", "RAIN", "E0", "ES0", "ET0", "WIND", "LAT", "LON"]
        columns = list(self.data.columns)
        columns.remove("DATE")
        new_weather_vars = []
        
        for w in all_weather_vars:
            if w in columns:
                new_weather_vars.append(w)
                columns.remove(w)
        new_output_vars = columns

        self.output_vars = new_output_vars
        self.weather_vars = new_weather_vars
        self.crop_config["weather_vars"] = new_weather_vars
        self.crop_config["output_vars"] = new_output_vars

        # Create model 
        self.true_model = GrapeModelFromFile(self.data, self.output_vars+self.weather_vars)
        self.model = GrapeModel(self.crop_config, drv=self.true_model.get_drv(self.day, self.weather_vars), params=args)

        return self.run_all()


    def _load_config_path(self, config_fpath:str):
        """
        Load the configuration file from .yaml
        """
        
        config = yaml.safe_load(open(config_fpath))
        agro = config["CropConfig"]
        crop_fpath = config["ModelConfig"]["config_fpath"]
        twin_config = config["ModelConfig"]

        # Create crop configuration
        crop = yaml.safe_load(open(os.path.join(crop_fpath, f"{agro['crop_name']}.yaml")))

        crop_config = crop["CropParameters"]["Varieties"][agro["variety_name"]]  

        for c in crop_config.keys():
            crop_config[c] = crop_config[c][0]

        for k,v in agro.items():
            crop_config[k] = v

        # Get start and end dates to update crop model
        if self.data is None:
            self.data = pd.read_csv(os.path.join(os.getcwd(), twin_config["digtwin_file"]), index_col=0)

        start_date = datetime.datetime.strptime(self.data["DATE"].iloc[0], '%Y-%m-%d')
        end_date = datetime.datetime.strptime(self.data["DATE"].iloc[-1], '%Y-%m-%d')

        crop_config["start_date"] = start_date
        crop_config["end_date"] = end_date

        # Update the output variables and weather variables
        all_weather_vars = ["IRRAD", "TMIN", "TMAX", "VAP", "TEMP", "RAIN", "E0", "ES0", "ET0", "WIND", "LAT", "LON"]
        columns = list(self.data.columns)
        columns.remove("DATE")
        new_weather_vars = []
        
        for w in all_weather_vars:
            if w in columns:
                new_weather_vars.append(w)
                columns.remove(w)
        new_output_vars = columns

        crop_config["weather_vars"] = new_weather_vars
        crop_config["output_vars"] = new_output_vars

        return self.data, crop_config

    def _load_config_dict(self, config:dict):
        """
        Load the configuration file from dictionary
        """
        
        agro = config["CropConfig"]
        crop_fpath = config["ModelConfig"]["crop_config"]
        twin_config = config["ModelConfig"]

        # Create crop configuration
        crop = yaml.safe_load(open(os.path.join(crop_fpath, f"{agro['crop_name']}.yaml")))

        crop_config = crop["CropParameters"]["Varieties"][agro["variety_name"]]  

        for c in crop_config.keys():
            crop_config[c] = crop_config[c][0]

        for k,v in agro.items():
            crop_config[k] = v

        # Get start and end dates to update crop model
        if self.data is None:
            self.data = pd.read_csv(os.path.join(os.getcwd(), twin_config["digtwin_file"]), index_col=0)

        start_date = datetime.datetime.strptime(self.data["DATE"].iloc[0], '%Y-%m-%d')
        end_date = datetime.datetime.strptime(self.data["DATE"].iloc[-1], '%Y-%m-%d')

        crop_config["start_date"] = start_date
        crop_config["end_date"] = end_date

        # Update the output variables and weather variables
        all_weather_vars = ["IRRAD", "TMIN", "TMAX", "VAP", "TEMP", "RAIN", "E0", "ES0", "ET0", "WIND", "LAT", "LON"]
        columns = list(self.data.columns)
        columns.remove("DATE")
        new_weather_vars = []
        
        for w in all_weather_vars:
            if w in columns:
                new_weather_vars.append(w)
                columns.remove(w)
        new_output_vars = columns

        crop_config["weather_vars"] = new_weather_vars
        crop_config["output_vars"] = new_output_vars

        return self.data, crop_config

    def get_param_dict(self):
        return self.model.get_param_dict()
    
    def save_model(self, path:str):
        """
        Save the model dictionary
        """
        self.model.save_model(path)

class GrapeModelFromFile():

    def __init__(self, data:pd.DataFrame, output_vars:list):
        """
        Initialize the grape model data with a dataframe
        """
        self.data = data[["DATE"]+output_vars]
        self.days_elapsed = 0

    def run(self, date:date):
        """
        Run the grape model for a day by getting the data from the dataframe
        """
        self.days_elapsed += 1
        output = self.data.iloc[self.days_elapsed]
        
        return output.to_list()
    
    def get_drv(self, date:date, weather_vars:list):
        """
        Get the driving variables as a weather data container
        """
        weather_output = dict(self.data.iloc[self.days_elapsed][weather_vars])
        
        drv = FileWeatherDataContainer(**weather_output)
        return drv
    
    def get_output(self):
        """
        Get the output on the current day
        """
        return self.data.iloc[self.days_elapsed].to_list()


class GrapeModel():
    
    def __init__(self, crop_config:dict, drv=None, params=None):
        """
        Initialize the grape model
        """
        self.model = GrapePhenologyEngine(config=crop_config, drv=drv, params=params)
    
    def run(self, date:date, drv):
        """
        Run a day in the model with the passed DRV
        """
        output = self.model.run(date=date, drv=drv)

        return output 
    
    def get_output(self):
        """
        Get the output on the current day
        """
        return self.model.get_output()
    
    def get_param_dict(self):
        return self.model.get_param_dict()
    
    def save_model(self, path:str):
        """
        Save the model dictionary
        """
        self.model.save_model(path)