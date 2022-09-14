import gym
import numpy as np
from collections import deque
import random
import re
import os
import sys
import time
import json
import itertools

# import stable_baselines3
from stable_baselines3 import PPO, A2C, DDPG, TD3
from stable_baselines3.common.utils import set_random_seed

from citylearn.citylearn import CityLearnEnv

import functools


class Constants:
    episodes = 3
    schema_path = 'data/citylearn_challenge_2022_phase_1/schema.json'


def action_space_to_dict(aspace):
    """ Only for box space """
    return {"high": aspace.high,
            "low": aspace.low,
            "shape": aspace.shape,
            "dtype": str(aspace.dtype)
            }


def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations}
    return obs_dict


import gym

# here we init the citylearn env
env = CityLearnEnv(schema=Constants.schema_path)

#### IMPORTANT
# here we choose the observation we want to take from the building env
# we divide observation that are specific to buildings (index_particular)
# and observation that are the same for all the buildings (index_commun)

index_commun = [0, 2, 19, 4, 8, 24]
index_particular = [20, 21, 22, 23]

normalization_value_commun = [12, 24, 2, 100, 100, 1]
normalization_value_particular = [5, 5, 5, 5]

len_tot_index = len(index_commun) + len(index_particular) * 5


## env wrapper for stable baselines
class EnvCityGym(gym.Env):
    """
    Env wrapper coming from the gym library.
    """

    def __init__(self, env):
        self.env = env

        # get the number of buildings
        self.num_buildings = len(env.action_space)

        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1] * self.num_buildings),
                                           high=np.array([1] * self.num_buildings), dtype=np.float32)

        # define the observation space
        self.observation_space = gym.spaces.Box(low=np.array([0] * len_tot_index), high=np.array([1] * len_tot_index),
                                                dtype=np.float32)

        # TO THINK : normalize the observation space

    def reset(self):
        obs_dict = env_reset(self.env)
        obs = self.env.reset()

        observation = self.get_observation(obs)

        return observation

    def get_observation(self, obs):
        """
        We retrieve new observation from the building observation to get a proper array of observation
        Basicly the observation array will be something like obs[0][index_commun] + obs[i][index_particular] for i in range(5)

        The first element of the new observation will be "commun observation" among all building like month / hour / carbon intensity / outdoor_dry_bulb_temperature_predicted_6h ...
        The next element of the new observation will be the concatenation of certain observation specific to buildings non_shiftable_load / solar_generation / ...
        """

        # we get the observation commun for each building (index_commun)
        observation_commun = [obs[0][i] / n for i, n in zip(index_commun, normalization_value_commun)]
        observation_particular = [[o[i] / n for i, n in zip(index_particular, normalization_value_particular)] for o in
                                  obs]

        observation_particular = list(itertools.chain(*observation_particular))
        # we concatenate the observation
        observation = observation_commun + observation_particular

        return observation

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        # reprocessing action
        action = [[act] for act in action]

        # we do a step in the environment
        obs, reward, done, info = self.env.step(action)

        observation = self.get_observation(obs)

        return observation, sum(reward), done, info

    def render(self, mode='human'):
        return self.env.render(mode)


# function to train the policy with PPO algorithm
def test_ppo():
    # Modify the petting zoo environment to make a custom observation space (return an array of value for each agent)

    # first we initialize the environment (petting zoo)
    env = CityLearnEnv(schema=Constants.schema_path)
    env = EnvCityGym(env)

    # we load the model
    model = PPO.load("ppo_citylearn")

    # we reset the environment
    obs = env.reset()

    nb_iter = 8000

    # loop on the number of iteration
    for i in range(nb_iter):
        # we get the action for each agent
        actions = []
        for agent in env.possible_agents:
            action, _states = model.predict(obs[agent], deterministic=True)

            actions.append(action)

        actions = {agent: action for agent, action in zip(env.possible_agents, actions)}

        # we do a step in the environment
        obs, rewards, dones, info = env.step(actions)

        # sometimes check the actions and rewards
        if i % 100 == 0:
            print("actions : ", actions)
            print("rewards : ", rewards)

    final_result = sum(env.citylearnenv.evaluate()) / 2

    print("final result : ", final_result)
    # launch as main

    return final_result


# function to train the policy with PPO algorithm
def train_ppo(train_timesteps):
    # first we initialize the environment (petting zoo)
    env = CityLearnEnv(schema=Constants.schema_path)
    env = EnvCityGym(env)

    env.reset()

    # Configure the algorithm

    # load model if exist
    # try:
    #     model = PPO.load("ppo_citylearn")
    # except:
    model = PPO('MlpPolicy', env, verbose=2, gamma=0.99)

    # Train the agent
    model.learn(total_timesteps=train_timesteps)

    model.save("ppo_citylearn")

    return model


# function to train the policy with PPO algorithm
def train_ddpg(train_timesteps):
    # first we initialize the environment (petting zoo)
    env = CityLearnEnv(schema=Constants.schema_path)
    env = EnvCityGym(env)

    env.reset()

    # Configure the algorithm

    # load model if exist
    # try:
    #     model = DDPG.load("ddpg_citylearn")
    # except:
    model = DDPG('MlpPolicy', env, verbose=2, gamma=0.99)

    # Train the agent
    model.learn(total_timesteps=train_timesteps)

    model.save("ddpg_citylearn")

    return model


# function to train the policy with PPO algorithm
def train_a2c(train_timesteps):
    # first we initialize the environment (petting zoo)
    env = CityLearnEnv(schema=Constants.schema_path)
    env = EnvCityGym(env)

    env.reset()

    # Configure the algorithm

    # load model if exist
    # try:
    #     model = A2C.load("a2c_citylearn")
    # except:
    model = A2C('MlpPolicy', env, verbose=2, gamma=0.99)

    # Train the agent
    model.learn(total_timesteps=train_timesteps)

    model.save("a2c_citylearn")

    return model


# function to train the policy with PPO algorithm
def train_td3(train_timesteps):
    # first we initialize the environment (petting zoo)
    env = CityLearnEnv(schema=Constants.schema_path)
    env = EnvCityGym(env)

    env.reset()

    # Configure the algorithm

    # load model if exist
    # try:
    #     model = TD3.load("td3_citylearn")
    # except:
    model = TD3('MlpPolicy', env, verbose=2, gamma=0.99)

    # Train the agent
    model.learn(total_timesteps=train_timesteps)

    model.save("td3_citylearn")

    return model


def evaluate_print_results(model, print_substeps=False):
    env = CityLearnEnv(schema=Constants.schema_path)
    env = EnvCityGym(env)

    obs = env.reset()

    # model = PPO.load("ppo_citylearn")

    nb_iter = 8750

    reward_tot = 0

    for i in range(nb_iter):

        action = model.predict(obs)[0]

        obs, rewards, dones, info = env.step(action)
        reward_tot += rewards

        if print_substeps:
            if i % 1000 == 0:
                print("actions : ", action)
                print("rewards : ", rewards)

    print(sum(env.env.evaluate()) / 2)
    print(reward_tot)


if __name__ == "__main__":
    t1_start = time.perf_counter()

    timesteps = 20000
    train_ppo(timesteps)
    train_ddpg(timesteps)
    train_a2c(timesteps)
    train_td3(timesteps)

    print("######PPO#######")

    ppo_start = time.perf_counter()
    model = PPO.load("ppo_citylearn")
    evaluate_print_results(model)
    ppo_stop = time.perf_counter()
    print("Elapsed time:", ppo_stop - ppo_start)

    print("######DDPG#######")

    ddpg_start = time.perf_counter()
    model = DDPG.load("ddpg_citylearn")
    evaluate_print_results(model)
    ddpg_stop = time.perf_counter()
    print("Elapsed time:", ddpg_stop - ddpg_start)

    print("######A2C#######")

    a2c_start = time.perf_counter()
    model = A2C.load("a2c_citylearn")
    evaluate_print_results(model)
    a2c_stop = time.perf_counter()
    print("Elapsed time:", a2c_stop - a2c_start)

    print("######TD3#######")

    td3_start = time.perf_counter()
    model = TD3.load("td3_citylearn")
    evaluate_print_results(model)
    td3_stop = time.perf_counter()
    print("Elapsed time:", td3_stop - td3_start)

    t1_stop = time.perf_counter()
    print("Total elapsed time:", t1_stop - t1_start)
