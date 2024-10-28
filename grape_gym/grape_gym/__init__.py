"""Entry point for WOFOST_Gym package. Handles imports and Gym Environment
registration.
"""

from gymnasium.envs.registration import register
from grape_gym import args
from grape_gym import utils
from grape_gym import exceptions

# Grape Environments
register(
    id='grape-lnpkw-v0',
    entry_point='grape_gym.envs.pheno_grape:Grape_Limited_NPKW_Env',
)
