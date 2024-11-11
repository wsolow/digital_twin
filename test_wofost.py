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

import grape_gym
import grape_gym.policies as policies
from utils import Args
import utils
import yaml


if __name__ == "__main__":

    args = tyro.cli(Args)

    env_id, env_kwargs = utils.get_gym_args(args)

    # Make the gym environment with wrappers
    env = gym.make(env_id, **env_kwargs)
    env = grape_gym.wrappers.NPKDictActionWrapper(env)
    env = grape_gym.wrappers.NPKDictObservationWrapper(env)
    
    # Set default policy for use

    policy = policies.No_Action(env)

    obs_arr = []
    obs, info = env.reset()
    term = False
    trunc = False
    obs_arr = []
    reward_arr = []

    # Run simulation and store data
    k = 0
    while not (term or trunc):
        action = policy(obs)
        next_obs, rewards, term, trunc, info = env.step(action)
        obs_arr.append(obs)
        reward_arr.append(rewards)
        obs = next_obs
        k+=1
        if (term or trunc):
            obs, info = env.reset()
            break
    all_obs = np.array([list(d.values()) for d in obs_arr])

    df = pd.DataFrame(data=np.array(all_obs), columns=env.get_output_vars())
    df.to_csv("data/dynamics_model.csv")

    all_vars = args.npk_args.output_vars + args.npk_args.forecast_length * args.npk_args.weather_vars
    print(f'SUCCESS in {args.env_id}')

    plt.plot(all_obs[:,6], all_obs[:,1])
    plt.xlabel('TEMP')
    plt.ylabel('TSUM')
    plt.show()

    for i in range(len(all_vars)):
        plt.figure(i)
        plt.title(all_vars[i])
        print(f'{i}: {all_vars[i]}')
        plt.plot(all_obs[ :, i+1])
        plt.xlim(0-10, all_obs.shape[0]+10) 
        plt.xlabel('Days')
    plt.show()


    



