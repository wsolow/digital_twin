"""Core API for environment wrappers for handcrafted policies and varying rewards."""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Box

from wofost_gym.envs.wofost_base import NPK_Env, Plant_NPK_Env, Harvest_NPK_Env

from wofost_gym.envs.wofost_annual import Limited_NPKW_Env
from wofost_gym.envs.wofost_annual import PP_Env
from wofost_gym.envs.wofost_annual import Limited_NPK_Env
from wofost_gym.envs.wofost_annual import Limited_N_Env
from wofost_gym.envs.wofost_annual import Limited_NW_Env
from wofost_gym.envs.wofost_annual import Limited_W_Env

from wofost_gym.envs.plant_annual import Plant_Limited_NPKW_Env
from wofost_gym.envs.plant_annual import Plant_PP_Env
from wofost_gym.envs.plant_annual import Plant_Limited_NPK_Env
from wofost_gym.envs.plant_annual import Plant_Limited_N_Env
from wofost_gym.envs.plant_annual import Plant_Limited_NW_Env
from wofost_gym.envs.plant_annual import Plant_Limited_W_Env

from wofost_gym.envs.harvest_annual import Harvest_Limited_NPKW_Env
from wofost_gym.envs.harvest_annual import Harvest_PP_Env
from wofost_gym.envs.harvest_annual import Harvest_Limited_NPK_Env
from wofost_gym.envs.harvest_annual import Harvest_Limited_N_Env
from wofost_gym.envs.harvest_annual import Harvest_Limited_NW_Env
from wofost_gym.envs.harvest_annual import Harvest_Limited_W_Env

from wofost_gym.envs.wofost_perennial import Perennial_Limited_NPKW_Env
from wofost_gym.envs.wofost_perennial import Perennial_PP_Env
from wofost_gym.envs.wofost_perennial import Perennial_Limited_NPK_Env
from wofost_gym.envs.wofost_perennial import Perennial_Limited_N_Env
from wofost_gym.envs.wofost_perennial import Perennial_Limited_NW_Env
from wofost_gym.envs.wofost_perennial import Perennial_Limited_W_Env

from wofost_gym.envs.plant_perennial import Perennial_Plant_Limited_NPKW_Env
from wofost_gym.envs.plant_perennial import Perennial_Plant_PP_Env
from wofost_gym.envs.plant_perennial import Perennial_Plant_Limited_NPK_Env
from wofost_gym.envs.plant_perennial import Perennial_Plant_Limited_N_Env
from wofost_gym.envs.plant_perennial import Perennial_Plant_Limited_NW_Env
from wofost_gym.envs.plant_perennial import Perennial_Plant_Limited_W_Env

from wofost_gym.envs.harvest_perennial import Perennial_Harvest_Limited_NPKW_Env
from wofost_gym.envs.harvest_perennial import Perennial_Harvest_PP_Env
from wofost_gym.envs.harvest_perennial import Perennial_Harvest_Limited_NPK_Env
from wofost_gym.envs.harvest_perennial import Perennial_Harvest_Limited_N_Env
from wofost_gym.envs.harvest_perennial import Perennial_Harvest_Limited_NW_Env
from wofost_gym.envs.harvest_perennial import Perennial_Harvest_Limited_W_Env

from wofost_gym import exceptions as exc

class NPKNaNToZeroWrapper(gym.ObservationWrapper):
    """Wraps the observation by converting nan's to zero. Good for use in some
    RL agents
    """
    def __init__(self, env: gym.Env):
        """Initialize the :class:`NPKNaNToZeroWrapper` wrapper with an environment.

        Casts all NaN's to zero

        Args: 
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.env = env

    def observation(self, obs):
        """Casts all NaNs in crop to zero
        
        Args:
            observation
        """
        return np.nan_to_num(obs, nan=0.0)

    def reset(self, **kwargs):
       """Reset the environment to the initial state specified by the 
        agromanagement, crop, and soil files.
        
        Args:
            **kwargs:
                year: year to reset enviroment to for weather
                location: (latitude, longitude). Location to set environment to"""
       obs, info = self.env.reset(**kwargs)
       return self.observation(obs), info


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

        # Harvesting environments
        if isinstance(self.env.unwrapped, Plant_NPK_Env):
            if isinstance(env, Plant_PP_Env) or \
                isinstance(self.env.unwrapped, Perennial_Plant_PP_Env): 
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1)})
            elif isinstance(self.env.unwrapped, Plant_Limited_NPK_Env) or \
                isinstance(self.env.unwrapped, Perennial_Plant_Limited_NPK_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "n": Discrete(self.env.unwrapped.num_fert),\
                                 "p": Discrete(self.env.unwrapped.num_fert),\
                                 "k": Discrete(self.env.unwrapped.num_fert)})
            elif isinstance(self.env.unwrapped, Plant_Limited_N_Env) or \
                isinstance(self.env.unwrapped, Perennial_Plant_Limited_N_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "n": Discrete(self.env.unwrapped.num_fert)})
            elif isinstance(self.env.unwrapped, Plant_Limited_NW_Env) or \
                isinstance(self.env.unwrapped, Perennial_Plant_Limited_NW_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "n": Discrete(self.env.unwrapped.num_fert),\
                                 "irrig": Discrete(self.env.unwrapped.num_irrig)})
            elif isinstance(self.env.unwrapped, Plant_Limited_W_Env) or \
                isinstance(self.env.unwrapped, Perennial_Plant_Limited_W_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "irrig": Discrete(self.env.unwrapped.num_irrig)})
            elif isinstance(self.env.unwrapped, Plant_Limited_NPKW_Env) or \
                isinstance(self.env.unwrapped, Perennial_Plant_Limited_NPKW_Env): 
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "n": Discrete(self.env.unwrapped.num_fert),\
                                 "p": Discrete(self.env.unwrapped.num_fert),\
                                 "k": Discrete(self.env.unwrapped.num_fert),\
                                 "irrig": Discrete(self.env.unwrapped.num_irrig)})
                
        elif isinstance(self.env.unwrapped, Harvest_NPK_Env):
            if isinstance(env, Harvest_PP_Env) or \
                isinstance(self.env.unwrapped, Perennial_Harvest_PP_Env): 
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1)})
            elif isinstance(self.env.unwrapped, Harvest_Limited_NPK_Env) or \
                isinstance(self.env.unwrapped, Perennial_Harvest_Limited_NPK_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "n": Discrete(self.env.unwrapped.num_fert),\
                                 "p": Discrete(self.env.unwrapped.num_fert),\
                                 "k": Discrete(self.env.unwrapped.num_fert)})
            elif isinstance(self.env.unwrapped, Harvest_Limited_N_Env) or \
                isinstance(self.env.unwrapped, Perennial_Harvest_Limited_N_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "n": Discrete(self.env.unwrapped.num_fert)})
            elif isinstance(self.env.unwrapped, Harvest_Limited_NW_Env) or \
                isinstance(self.env.unwrapped, Perennial_Harvest_Limited_NW_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "n": Discrete(self.env.unwrapped.num_fert),\
                                 "irrig": Discrete(self.env.unwrapped.num_irrig)})
            elif isinstance(self.env.unwrapped, Harvest_Limited_W_Env) or \
                isinstance(self.env.unwrapped, Perennial_Harvest_Limited_W_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "irrig": Discrete(self.env.unwrapped.num_irrig)})
            elif isinstance(self.env.unwrapped, Harvest_Limited_NPKW_Env) or \
                isinstance(self.env.unwrapped, Perennial_Harvest_Limited_NPKW_Env): 
                self.action_space = gym.spaces.Dict({"null": Discrete(1), \
                                 "plant": Discrete(1), "harvest": Discrete(1), \
                                 "n": Discrete(self.env.unwrapped.num_fert),\
                                 "p": Discrete(self.env.unwrapped.num_fert),\
                                 "k": Discrete(self.env.unwrapped.num_fert),\
                                 "irrig": Discrete(self.env.unwrapped.num_irrig)})
        # Default environments
        else: 
            if isinstance(self.env.unwrapped, PP_Env) or \
                isinstance(self.env.unwrapped, Perennial_PP_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1), "n": Discrete(1)})
            elif isinstance(self.env.unwrapped, Limited_NPK_Env) or \
                isinstance(self.env.unwrapped, Perennial_Limited_NPK_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1),\
                                 "n": Discrete(self.env.unwrapped.num_fert),\
                                 "p": Discrete(self.env.unwrapped.num_fert),\
                                 "k": Discrete(self.env.unwrapped.num_fert)})
            elif isinstance(self.env.unwrapped, Limited_N_Env) or \
                isinstance(self.env.unwrapped, Perennial_Limited_N_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1),\
                                 "n": Discrete(self.env.unwrapped.num_fert)})
            elif isinstance(self.env.unwrapped, Limited_NW_Env) or \
                isinstance(self.env.unwrapped, Perennial_Limited_NW_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1),\
                                 "n": Discrete(self.env.unwrapped.num_fert),\
                                 "irrig": Discrete(self.env.unwrapped.num_irrig)})
            elif isinstance(self.env.unwrapped, Limited_W_Env) or \
                isinstance(self.env.unwrapped, Perennial_Limited_W_Env):
                self.action_space = gym.spaces.Dict({"null": Discrete(1),\
                                 "irrig": Discrete(self.env.unwrapped.num_irrig)})
            elif isinstance(self.env.unwrapped, Limited_NPKW_Env) or \
                isinstance(self.env.unwrapped, Perennial_Limited_NPKW_Env): 
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

class RewardWrapper(gym.Wrapper):
    """ Abstract class for all reward wrappers
    
    Given how the reward wrapper functions, it must be applied BEFORE any
    observation or action wrappers. 
    
    This _validate() function ensures that is the case and will throw and error
    otherwise 
    """
    def __init__(self, env: gym.Env):
        """Initialize the :class:`RewardWrapper` wrapper with an environment.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self._validate(env)
        self.env = env

    
    def _validate(self, env: gym.Env):
        """Validates that the environment is not wrapped with an Observation or 
        Action Wrapper
        
        Args: 
            env: The environment to check
        """
        if isinstance(env, gym.ActionWrapper) or isinstance(env, gym.ObservationWrapper):
            msg = "Cannot wrap an Action or Observation Wrapped Environment. Use reward wrapper before Action or Observation Wrapper."
            raise exc.WOFOSTGymError(msg)
        if isinstance(env, RewardWrapper):
            msg = "Cannot wrap environment with another reward wrapper."
            raise exc.AcWOFOSTGymError(msg)

    def reset(self, **kwargs):
       """Reset the environment to the initial state specified by the 
        agromanagement, crop, and soil files.
        
        Args:
            **kwargs:
                year: year to reset enviroment to for weather
                location: (latitude, longitude). Location to set environment to"""
       return self.env.reset(**kwargs)
 
class RewardFertilizationCostWrapper(RewardWrapper):
    """ Modifies the reward to be a function of how much fertilization and irrigation
    is applied
    """
    def __init__(self, env: gym.Env, cost: float=10):
        """Initialize the :class:`RewardFertilizationCostWrapper` wrapper with an environment.

        Args: 
            env: The environment to apply the wrapper
            cost: The cost scaler to be used to scale the reward penalty 
        """
        super().__init__(env)
        self.env = env
        self.cost = cost
    
    def _get_reward(self, output: dict, act_tuple:tuple):
        """Gets the reward as a penalty based on the amount of NPK/Water applied
        
        Args:
            output: dict     - output from model
            act_tuple: tuple -  NPK/Water amounts"""
        if self.env.unwrapped.NUM_ACT == 6:
            reward = output.iloc[-1]['WSO'] - \
                            (np.sum(10 * np.array([act_tuple])))
        elif self.env.unwrapped.NUM_ACT == 4: 
            reward = output.iloc[-1]['WSO'] - \
                            (np.sum(10 * np.array([act_tuple[2:]])))
        return reward
         
class RewardFertilizationThresholdWrapper(RewardWrapper):
    """ Modifies the reward to be a function with high penalties for if a 
     threshold is crossed during fertilization or irrigation
    """
    def __init__(self, env: gym.Env, max_n: float=np.inf, max_p: float=np.inf, max_k: float=np.inf, max_w: float=np.inf):
        """Initialize the :class:`RewardFertilizationThresholdWrapper` wrapper with an environment.

        Args: 
            env: The environment to apply the wrapper
            max_n: Nitrogen threshold
            max_p: Phosphorous threshold
            max_k: Potassium threshold
            max_w: Irrigation threshold
        """
        super().__init__(env)
        self.env = env

        # Thresholds for nutrient application
        self.max_n = max_n
        self.max_p = max_p
        self.max_k = max_k
        self.max_w = max_w

    def step(self, action):
        """Run one timestep of the environment's dynamics.

        Sends action to the WOFOST model and recieves the resulting observation
        which is then processed to the _get_reward() function and _process_output()
        function for a reward and observation

        Args:
            action: integer
        """
        # Send action signal to model and run model
        act_tuple = self.env.unwrapped._take_action(action)
        output = self.env.unwrapped._run_simulation()

        observation = self.env.unwrapped._process_output(output)
        
        reward = self._get_reward(output, act_tuple) 
        
        # Terminate based on site end date
        terminate = self.env.unwrapped.date >= self.env.unwrapped.site_end_date
        # Truncate based on crop finishing
        truncation = output.iloc[-1]['FIN'] == 1.0

        self.env.unwrapped._log(output.iloc[-1]['WSO'], act_tuple, reward)
        return observation, reward, terminate, truncation, self.env.unwrapped.log
    
    def _get_reward(self, output, act_tuple):
        """Convert the reward by applying a high penalty if a fertilization
        threshold is crossed
        
        Args:
            output     - of the simulator
            act_tuple  - amount of NPK/Water applied
        """
        if output.iloc[-1]['TOTN'] > self.max_n and act_tuple[self.env.unwrapped.N] > 0:
            return -1e4 * act_tuple[self.env.unwrapped.N]
        if output.iloc[-1]['TOTP'] > self.max_p and act_tuple[self.env.unwrapped.P] > 0:
            return -1e4 * act_tuple[self.env.unwrapped.P]
        if output.iloc[-1]['TOTK'] > self.max_k and act_tuple[self.env.unwrapped.K] > 0:
            return -1e4 * act_tuple[self.env.unwrapped.K]
        if output.iloc[-1]['TOTIRRIG'] > self.max_w and act_tuple[self.env.unwrapped.I] > 0:
            return -1e4 * act_tuple[self.env.unwrapped.I]
        return np.nan_to_num(output.iloc[-1]['WSO'])
    

