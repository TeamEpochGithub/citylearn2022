import gym
import numpy as np
import random
import re
import os
import sys
import time
import json
import itertools
from citylearn.citylearn import CityLearnEnv

class EnvCityGym(gym.Env):
    """
    Env wrapper coming from the gym library.
    """
    def __init__(self):

        self.index_com = [0, 2, 19, 4, 8, 24]
        self.index_part = [20, 21, 22, 23]
        self.normalization_value_com = [12, 24, 2, 100, 100, 1]
        self.normalization_value_part = [5, 5, 5, 5]
        self.len_tot_index = len(self.index_com) + len(self.index_part) * 5

        #self.env = CityLearnEnv(schema=f"{os.path.dirname(os.path.realpath(__file__))}/data/citylearn_challenge_2022_phase_1/schema.json")
        self.env = CityLearnEnv(schema=r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\schema.json")

        # get the number of buildings
        self.num_buildings = len(self.env.action_space)

        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1] * self.num_buildings),
                                           high=np.array([1] * self.num_buildings), dtype=np.float32)

        # define the observation space
        # self.observation_space = gym.spaces.Box(low=np.array([0] * self.len_tot_index), high=np.array([1] * self.len_tot_index),
        #                                         dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=np.array([-2.5] * self.len_tot_index),
                                                high=np.array([2.5] * self.len_tot_index),
                                                dtype=np.float32)

        # TO THINK : normalize the observation space

    def reset(self):
        obs_dict = self.env_reset(self.env)
        obs = self.env.reset()

        observation = self.get_observation(obs)

        return (np.array(observation, dtype=np.float32))
        #return np.array(observation)

    def action_space_to_dict(self, aspace):
        """ Only for box space """
        return {"high": aspace.high,
                "low": aspace.low,
                "shape": aspace.shape,
                "dtype": str(aspace.dtype)
                }

    def env_reset(self, env):
        observations = env.reset()
        action_space = env.action_space
        observation_space = env.observation_space
        building_info = env.get_building_information()
        building_info = list(building_info.values())

        action_space_dicts = [self.action_space_to_dict(asp) for asp in action_space]
        observation_space_dicts = [self.action_space_to_dict(osp) for osp in observation_space]
        obs_dict = {"action_space": action_space_dicts,
                    "observation_space": observation_space_dicts,
                    "building_info": building_info,
                    "observation": observations}
        return obs_dict

    def get_observation(self, obs):
        """
        We retrieve new observation from the building observation to get a proper array of observation
        Basicly the observation array will be something like obs[0][index_common] + obs[i][index_particular] for i in range(5)

        The first element of the new observation will be "common observation" among all building like month / hour / carbon intensity / outdoor_dry_bulb_temperature_predicted_6h ...
        The next element of the new observation will be the concatenation of certain observation specific to buildings non_shiftable_load / solar_generation / ...
        """

        # we get the observation common for each building (index_common)
        observation_common = [obs[0][i] / n for i, n in zip(self.index_com, self.normalization_value_com)]
        observation_particular = [[o[i] / n for i, n in zip(self.index_part, self.normalization_value_part)] for o in obs]

        observation_particular = list(itertools.chain(*observation_particular))
        # we concatenate the observation
        observation = observation_common + observation_particular

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

        #return observation, sum(reward), done, info
        #sys.exit()
        return np.array(observation, dtype=np.float32), sum(reward), done, info


    def render(self, mode='human'):
        return self.env.render(mode)