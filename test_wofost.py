"""File for testing the installation and setup of the WOFOST Gym Environment
with a few simple plots for output 

Written by: Will Solow, 2024
"""

import gymnasium as gym
import numpy as np
import pandas as pd
import tyro
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
from datetime import datetime

import wofost_gym
import wofost_gym.policies as policies
from wofost_gym.envs.wofost_base import NPK_Env, Plant_NPK_Env, Harvest_NPK_Env
from utils import Args
import utils
import yaml


if __name__ == "__main__":

    args = tyro.cli(Args)

    env_id, env_kwargs = utils.get_gym_args(args)

    # Make the gym environment with wrappers
    env = gym.make(env_id, **env_kwargs)
    env = wofost_gym.wrappers.NPKDictActionWrapper(env)
    env = wofost_gym.wrappers.NPKDictObservationWrapper(env)
    
    # Set default policy for use
    if isinstance(env.unwrapped, Harvest_NPK_Env):
        policy = policies.No_Action_Harvest(env)
    elif isinstance(env.unwrapped, Plant_NPK_Env):
        policy = policies.No_Action_Plant(env)
    else:
        policy = policies.Interval_N(env, amount=0, interval=1)

    obs_arr = []
    obs, info = env.reset()
    term = False
    trunc = False
    obs_arr = []
    reward_arr = []

    # Run simulation and store data
    k = 0
    while not term:
        action = policy(obs)
        next_obs, rewards, term, trunc, info = env.step(action)
        obs_arr.append(obs)
        reward_arr.append(rewards)
        obs = next_obs
        k+=1
        if term or trunc:
            obs, info = env.reset()
            break
    all_obs = np.array([list(d.values()) for d in obs_arr])


    df = pd.DataFrame(data=np.array(all_obs), columns=env.get_output_vars())
    df.to_csv("data/below_n.csv")


    all_vars = args.npk_args.output_vars + args.npk_args.forecast_length * args.npk_args.weather_vars
    print(f'SUCCESS in {args.env_id}')


    for i in range(len(all_vars)):
        plt.figure(i)
        plt.title(all_vars[i])
        plt.plot(all_obs[ :, i])
        plt.xlim(0-10, all_obs.shape[0]+10) 
        plt.xlabel('Days')
    plt.show()


    



