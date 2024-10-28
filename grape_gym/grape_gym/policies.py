"""Policy class containing hand crafted policies for various WOFOST Gym Environments

Written by: Will Solow, 2024
"""
import gymnasium as gym
from grape_gym.exceptions import PolicyException
from abc import abstractmethod

class Policy:
    """Abstract Policy class containing the validate method

    Requires that the Gym Environment is wrapped with the NPKDiscreteObservationWrapper
    to ensure easy policy specification, specifically for policies that depend on
    the state. 
    """
    required_vars = list

    def __init__(self, env: gym.Env, required_vars: list=[]):
        """Initialize the :class:`Policy`.

        Args: 
            env: The Gymnasium Environment
            required_vars: list of required state space variables 
        """
        self.required_vars = required_vars
        self.env = env

        self._validate()
    
    def __call__(self, obs:dict):
        """Calls the _get_action() method. 
        
        Helper method so that the HandCrafted Policies share same call 
        structure to RL Agents. So, RL Agent Policy or Handcrafted Policy can be 
        returned from a function and called even without knowing what type of policy 
        it is."""
        return self._get_action(obs)
    
    def _validate(self):
        """Check that the policy is valid given the observation space and that
        the environment is wrapped with the NPKDictObservationWrapper
        """
        obs = self.env.observation_space.sample()
        if isinstance(obs, dict):
            for key in self.required_vars:
                if not key in list(obs.keys()):
                    msg = f"Required observation `{key}` for policy is not in inputted observation "
                    raise PolicyException(msg)
        else:
            msg = "Observation Space is not of type `Dict`. Wrap Environment with NPKDictObservationWrapper before proceeding."
            raise PolicyException(msg)
        
        action = self.env.action_space.sample()

        if not isinstance(action, dict):
            msg = "Action Space is not of type `Dict`. Wrap Environment with NPKDictActionWrapper before proceeding."
            raise PolicyException(msg)

    @abstractmethod     
    def _get_action(self, obs:dict):
        """Return the action for the environment to take
        """
        msg = "Policy Subclass should implement this"
        raise NotImplementedError(msg) 
    
    @abstractmethod
    def __str__(self):
        """
        Returns the string representation
        """

class No_Action(Policy):
    """Default policy performing no irrigation or fertilization actions
    """
    required_vars = []

    def __init__(self, env: gym.Env):
        """Initialize the :class:`No_Action`.

        Args: 
            env: The Gymnasium Environment
        """
        super().__init__(env, required_vars=self.required_vars)

    def _get_action(self, obs):
        return {'n': 0, 'p': 0, 'k': 0, 'irrig':0 }
    
    def __str__(self):
        """
        Returns a human readable string
        """
        return f"no_action"