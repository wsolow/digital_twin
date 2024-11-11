"""
The Engine class in control of running the grape phenology model 
"""
import datetime
from datetime import date
from traitlets_pcse import Instance, HasTraits
from .util import data_loader, set_model_params
from .grape_phenology import Grape_Phenology
from .nasapower import NASAPowerWeatherDataProvider, WeatherDataProvider, WeatherDataContainer
import pandas as pd

class GrapePhenologyEngine(HasTraits):
    """Convenience class for running WOFOST8.0 nutrient and water-limited production

    :param parameterprovider: A ParameterProvider instance providing all parameter values
    :param weatherdataprovider: A WeatherDataProvider object
    :param agromanagement: Agromanagement data
    """
    # sub components for simulation
    model = Instance(Grape_Phenology)
    weatherdataprovider = Instance(WeatherDataProvider,allow_none=True)
    drv = None
    day = Instance(date)

    def __init__(self, config:dict=None, drv=None, config_fpath:str=None, crop_fpath:str=None, args:dict=None):
        """
        Initialize GrapePhenologyEngine Class Class
        """
        # Get model configuration
        if config is None:
            self.config = data_loader(config_fpath, crop_fpath)
        else:
            self.config = config
        
        self.start_date = self.config["start_date"]
        self.end_date = self.config["end_date"]
        self.day = self.start_date

        # Driving variables
        if drv is None:
            self.weatherdataprovider = NASAPowerWeatherDataProvider(self.config["latitude"], self.config["longitude"])
            self.drv = self.weatherdataprovider(self.day)
        else:
            self.weatherdataprovider = None
            self.drv = drv

        # initialize model and set params as needed
        self.model = Grape_Phenology(self.start_date, self.config)
        if args is not None:
            set_model_params(self.model, args)

        # Output variables
        self.output_vars = self.config["output_vars"]
        if self.output_vars == None:
            self.output_vars = self.model.get_output_vars()
        self.weather_vars = self.config["weather_vars"]

        # Calculate initial rates
        self.calc_rates(self.day, self.drv)

    def calc_rates(self, day:date, drv:WeatherDataContainer):
        """
        Calculate the rates for computing rate of state change
        """
        self.model.calc_rates(day, drv)


    def integrate(self, day:date, delt:float):
        """
        Integrate rates with states based on time change (delta)
        """
        self.model.integrate(day, delt)

        # Set all rate variables to zero
        self.zerofy()

    def _run(self, drv=None, date:datetime.date=None, delt=1):
        """
        Make one time step of the simulation.
        """
        # Update day
        if date is None:
            self.day += datetime.timedelta(days=delt)
        else:
            self.day = date
        
        # Get driving variables
        if drv is None:
            self.drv = self.weatherdataprovider(self.day)
        else: 
            self.drv = drv
        # State integration
        self.integrate(self.day, delt)

        # Rate calculation
        self.calc_rates(self.day, self.drv)


    def run(self, date:datetime.date=None, drv=None, days:int=1):
        """
        Advances the system state with given number of days
        """

        days_done = 0
        while (days_done < days):
            days_done += 1
            self._run(drv=drv, date=date)

        return self.get_output()

    def run_all(self):
        """
        Run a simulation through termination
        """

        output = [self.get_crop_output()+self.get_weather_output()]

        while self.day < self.end_date:
            daily_output = self.run()
            output.append(daily_output)

        return pd.DataFrame(output, columns=["DATE"]+self.output_vars+self.weather_vars)

    def get_crop_output(self):
        """
        Return the output of the model
        """
        return [self.day] + self.model.get_output(vars=self.output_vars) 

    def get_weather_output(self):
        """
        Get the weather output for the day
        """
        weather = []
        for v in self.weather_vars:
            weather.append(getattr(self.drv, v))
        return weather

    def get_output(self):
        """
        Get all crop and weather output
        """
        return self.get_crop_output()+self.get_weather_output()

    def zerofy(self):
        """
        Zero out all the rates
        """
        self.model.rates.zerofy()

    def get_param_dict(self):
        """
        Get the parameter dictionary 
        """
        return self.model.get_param_dict()
    
    def save_model(self, path:str):
        """
        Save the model as a dictionary
        """
        self.model.save_model(path)


