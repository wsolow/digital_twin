"""This module wraps the soil components for water and nutrients so that they 
run jointly within the same model.
Allard de Wit (allard.dewit@wur.nl), September 2020
Modified by Will Solow, 2024
"""
from datetime import date 

from ..utils.traitlets import Instance
from ..nasapower import WeatherDataProvider
from ..base import SimulationObject, VariableKiosk
from .classic_waterbalance import WaterbalanceFD
from .npk_soil_dynamics import NPK_Soil_Dynamics

class BaseSoilModuleWrapper(SimulationObject):
    """Base Soil Module Wrapper
    """
    WaterbalanceFD = Instance(SimulationObject)
    NPK_Soil_Dynamics = Instance(SimulationObject)

    def initialize(self, day:date , kiosk:VariableKiosk, parvalues:dict):
        msg = "`initialize` method not yet implemented on %s" % self.__class__.__name__
        raise NotImplementedError(msg)
    
    def calc_rates(self, day:date, drv:WeatherDataProvider):
        """Calculate state rates
        """
        self.WaterbalanceFD.calc_rates(day, drv)
        self.NPK_Soil_Dynamics.calc_rates(day, drv)

    def integrate(self, day:date, delt:float=1.0):
        """Integrate state rates
        """
        self.WaterbalanceFD.integrate(day, delt)
        self.NPK_Soil_Dynamics.integrate(day, delt)

class SoilModuleWrapper_LNPKW(BaseSoilModuleWrapper):
    """This wraps the soil water balance for free drainage conditions and NPK balance
    for production conditions limited by both soil water and NPK.
    """

    def initialize(self, day:date, kiosk:VariableKiosk, parvalues:dict):
        """
        :param day: start date of the simulation
        :param kiosk: variable kiosk of this PCSE instance
        :param parvalues: dictionary with parameter key/value pairs
        """
        self.WaterbalanceFD = WaterbalanceFD(day, kiosk, parvalues)
        self.NPK_Soil_Dynamics = NPK_Soil_Dynamics(day, kiosk, parvalues)
