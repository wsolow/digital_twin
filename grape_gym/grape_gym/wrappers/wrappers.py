"""Core API for environment wrappers for handcrafted policies and varying rewards."""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box

from grape_gym.envs.pheno_base import NPK_Env
from grape_gym.envs.pheno_grape import Grape_Limited_NPKW_Env
from grape_gym import exceptions as exc

class NPKDictObservationWrapper(gym.ObservationWrapper):
    """Wraps the observation in a dictionary for easy access to variables
    without relying on direct indexing
    """
    def __init__(self, env: gym.Env):
        """Initialize the :class:`NPKDictObservationWrapper` wrapper with an environment.

        Handles extended weather forecasts by appending an _i to all weather
        variables, where {i} is the day. 

        Args: 
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.env = env
        self.output_vars = self.env.unwrapped.output_vars
        self.forecast_vars = []

        self.weather_vars = self.env.unwrapped.weather_vars
        if self.env.unwrapped.forecast_length > 1:
            self.forecast_vars = []
            for i in range(1, self.env.unwrapped.forecast_length):
                self.forecast_vars += [s + f"_{i+1}" for s in self.weather_vars]
        self.forecast_vars += self.weather_vars 

        output_dict = [(ov, Box(low=-np.inf, high=np.inf,shape=(1,))) for ov in self.output_vars]
        weather_dict = [(wv, Box(low=-np.inf, high=np.inf,shape=(1,))) for wv in self.output_vars]

        self.observation_space = Dict(dict(output_dict+weather_dict+\
                                           [("DAYS", Box(low=-np.inf, high=np.inf,shape=(1,)))]))

    def get_output_vars(self):
        """Return a list of the output vars"""
        return self.output_vars + self.weather_vars + ["DAYS"]
    
    def observation(self, obs):
        """Puts the outputted variables in a dictionary.

        Note that the dictionary must be in order of the variables. This will not
        be a problem if the output is taken directly from the environment which
        already enforces order.
        
        Args:
            observation
        """
        keys = self.output_vars + self.forecast_vars + ["DAYS"]
        return dict([(keys[i], obs[i]) for i in range(len(keys))])

    def reset(self, **kwargs):
       """Reset the environment to the initial state specified by the 
        agromanagement, crop, and soil files.
        
        Args:
            **kwargs:
                year: year to reset enviroment to for weather
                location: (latitude, longitude). Location to set environment to"""
       obs, info = self.env.reset(**kwargs)
       return self.observation(obs), info

class NPKDictActionWrapper(gym.ActionWrapper):
    """Converts a wrapped action to an action interpretable by the simulator.
    
    This wrapper is necessary for all provided hand-crafted policies which return
    an action as a dictionary. See policies.py for more information. 
    """
    def __init__(self, env: gym.Env):
        """Initialize the :class:`NPKDictActionWrapper` wrapper with an environment.

        Args: 
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.env = env
        self.num_fert = self.env.unwrapped.num_fert
        self.num_irrig = self.env.unwrapped.num_irrig

        # Base Grape Model
        if isinstance(self.env.unwrapped, Grape_Limited_NPKW_Env): 
            self.action_space = gym.spaces.Dict({"null": Discrete(1),\
                                "n": Discrete(self.env.unwrapped.num_fert),\
                                "p": Discrete(self.env.unwrapped.num_fert),\
                                "k": Discrete(self.env.unwrapped.num_fert),\
                                "irrig": Discrete(self.env.unwrapped.num_irrig)})

    def action(self, act: dict):
        """Converts the dicionary action to an integer to be pased to the base
        environment.
        
        Args:
            action
        """
        if not isinstance(act, dict):
            msg = "Action must be of dictionary type. See README for more information"
            raise exc.ception(msg)
        else: 
            act_vals = list(act.values())
            for v in act_vals:
                if not isinstance(v, int):
                    msg = "Action value must be of type int"
                    raise exc.ActionException(msg)
            if len(np.nonzero(act_vals)[0]) > 1:
                msg = "More than one non-zero action value for policy"
                raise exc.ActionException(msg)
            # If no actions specified, assume that we mean the null action
            if len(np.nonzero(act_vals)[0]) == 0:
                return 0
        
        if not "n" in act.keys():
            msg = "Nitrogen action \'n\' not included in action dictionary keys"
            raise exc.ActionException(msg)
        if not "p" in act.keys():
            msg = "Phosphorous action \'p\' not included in action dictionary keys"
            raise exc.ActionException(msg)
        if not "k" in act.keys():
            msg = "Potassium action \'k\' not included in action dictionary keys"
            raise exc.ActionException(msg)
        if not "irrig" in act.keys():
            msg = "Irrigation action \'irrig\' not included in action dictionary keys"
            raise exc.ActionException(msg)

        # Planting Single Year environments
        if isinstance(self.env.unwrapped, Plant_NPK_Env):
            # Check for planting and harvesting actions
            if not "plant" in act.keys():
                msg = "\'plant\' not included in action dictionary keys"
                raise exc.ActionException(msg)
            if not "harvest" in act.keys():
                msg = "\'harvest\' not included in action dictionary keys"
                raise exc.ActionException(msg)
            if len(act.keys()) != self.env.unwrapped.NUM_ACT:
                msg = "Incorrect action dictionary specification"
                raise exc.ActionException(msg)
            
            # Set the offsets to support converting to the correct action
            offsets = [1,1,self.num_fert,self.num_fert,self.num_fert,self.num_irrig]
            act_values = [act["plant"],act["harvest"],act["n"],act["p"],act["k"],act["irrig"]]
            offset_flags = np.zeros(self.env.unwrapped.NUM_ACT)
            offset_flags[:np.nonzero(act_values)[0][0]] = 1

        # Harvesting Single Year environments
        elif isinstance(self.env.unwrapped, Harvest_NPK_Env):
            # Check for harvesting actions
            if not "harvest" in act.keys():
                msg = "\'harvest\' not included in action dictionary keys"
                raise exc.ActionException(msg)
            if len(act.keys()) != self.env.unwrapped.NUM_ACT:
                msg = "Incorrect action dictionary specification"
                raise exc.ActionException(msg)
            
            # Set the offsets to support converting to the correct action
            offsets = [1,self.num_fert,self.num_fert,self.num_fert,self.num_irrig]
            act_values = [act["harvest"],act["n"],act["p"],act["k"],act["irrig"]]
            offset_flags = np.zeros(self.env.unwrapped.NUM_ACT)
            offset_flags[:np.nonzero(act_values)[0][0]] = 1

        # Default environments
        else: 
            if len(act.keys()) != self.env.unwrapped.NUM_ACT:
                msg = "Incorrect action dictionary specification"
                raise exc.ActionException(msg)
            # Set the offsets to support converting to the correct action
            offsets = [self.num_fert,self.num_fert,self.num_fert,self.num_irrig]
            act_values = [act["n"],act["p"],act["k"],act["irrig"]]
            offset_flags = np.zeros(self.env.unwrapped.NUM_ACT)
            offset_flags[:np.nonzero(act_values)[0][0]] = 1
            
        return np.sum(offsets*offset_flags) + act_values[np.nonzero(act_values)[0][0]] 
            
    def reset(self, **kwargs):
       """Reset the environment to the initial state specified by the 
        agromanagement, crop, and soil files.
        
        Args:
            **kwargs:
                year: year to reset enviroment to for weather
                location: (latitude, longitude). Location to set environment to"""
       return self.env.reset(**kwargs)

