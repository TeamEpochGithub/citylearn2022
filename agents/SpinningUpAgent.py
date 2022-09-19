import random
import numpy as np
import pandas as pd
import os
from bisect import bisect_left
import sys
import torch
import os.path as osp
import itertools

# Split action space into discrete regions.
# Split environment space into discrete regions.

# Environment space: Electricity cost, Carbon cost, Hour, Total solar energy
# Array values: EC: 24, CC: 19, H: 2, TSE: 11 + 15,





class BasicPPOAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """
    def __init__(self):
        self.action_space = {}
        self.directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        self.ac = torch.load(osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),r'SpinningUp\\data\\ppo\\ppo_s0\\pyt_save\\model.pt'))

        self.index_com = [0, 2, 19, 4, 8, 24]
        self.index_part = [20, 21, 22, 23]
        self.normalization_value_com = [12, 24, 2, 100, 100, 1]
        self.normalization_value_part = [5, 5, 5, 5]

        self.len_tot_index = len(self.index_com) + len(self.index_part) * 5

    def set_action_space(self, action_space):
        self.action_space = action_space


    def compute_action(self, observations, building_count):
        """Get observation return action"""

        return self.ppo_policy(observations, building_count)

    def ppo_policy(self, observations, building_count):

        observation_common = [observations[0][i] / n for i, n in zip(self.index_com, self.normalization_value_com)]
        observation_particular = [[o[i] / n for i, n in zip(self.index_part, self.normalization_value_part)] for o in observations]

        observation_particular = list(itertools.chain(*observation_particular))
        transformed_observation = observation_common + observation_particular

        actions = self.ac.act(torch.as_tensor(transformed_observation, dtype=torch.float32))

        action_list = []

        for a in actions:
            action_list.append(np.array([a], dtype=np.float32))

        return action_list
