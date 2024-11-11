"""Main API for default WOFOST Gym environments with actions for NPK and water
application.

Used for grape crop simulations for digital twin modelling
"""

import gymnasium as gym

from grape_gym.args import NPK_Args
from grape_gym import utils
from grape_gym.envs.pheno_base import NPK_Env

import grape_model as gp
from grape_model.soil.soil_wrappers import SoilModuleWrapper_LNPKW
from grape_model.crop.phenology import Grape_Phenology
from grape_model.agromanager import AgroManagerPerennial

class Grape_Limited_NPKW_Env(NPK_Env):
    """Simulates crop growth under NPK and water limited conditions
    """
    config = utils.make_config(soil=SoilModuleWrapper_LNPKW, crop=Grape_Phenology, \
                               agro=AgroManagerPerennial)
    def __init__(self, args: NPK_Args, base_fpath: str, agro_fpath:str, \
                 site_fpath:str, crop_fpath: str):
        """Initialize the :class:`Limited_NPKW_Env`.

        Args: 
            NPK_Args: The environment parameterization
        """
        super().__init__(args, base_fpath, agro_fpath, site_fpath, crop_fpath, \
                         config=self.config)

        self.action_space = gym.spaces.Discrete(1+3*self.num_fert+self.num_irrig)
    
    def _take_action(self, action: int):
        """Controls sending fertilization and irrigation signals to the model. 
        Converts the integer action to a signal and amount of NPK/Water to be applied.
        
        Args:
            action
        """
        n_amount = 0
        p_amount = 0
        k_amount = 0
        irrig_amount = 0

        # Null action
        if action == 0: 
            return (n_amount, p_amount, k_amount, irrig_amount)
        
        # Irrigation action
        if action >= 3 * self.num_fert+1:
            i_amount = action - (3 * self.num_fert)
            i_amount *= self.irrig_amount
            self.model._send_signal(signal=gp.signals.irrigate, amount=i_amount, \
                                    efficiency=self.irrig_effec)
            return (n_amount, p_amount, k_amount, irrig_amount)
        
        # Fertilizaiton action, correct for null action
        if (action-1) // self.num_fert == 0:
            n_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=gp.signals.apply_npk, \
                                    N_amount=n_amount, N_recovery=self.n_recovery)
        elif (action-1) // self.num_fert == 1:
            p_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=gp.signals.apply_npk, \
                                    P_amount=p_amount, P_recovery=self.p_recovery)
        elif (action-1) // self.num_fert == 2:
            k_amount = self.fert_amount * (( (action-1) % self.num_fert)+1) 
            self.model._send_signal(signal=gp.signals.apply_npk, \
                                        K_amount=k_amount, K_recovery=self.k_recovery)
            
        return (n_amount, p_amount, k_amount, irrig_amount)

