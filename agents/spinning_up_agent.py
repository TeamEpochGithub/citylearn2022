import sys

import numpy as np
import torch
import os.path as osp
import itertools
from traineval.training.spinningup.data.ppo.ppo_s0 import pyt_save
from traineval.training.spinningup import data as saved_models

class BasicPPOAgent:

    def __init__(self, environment_arguments, model_type:str, model_seed:int, model_iteration:int):
        self.action_space = {}
        model_path = model_type + '\\' + model_type + '_s' + str(model_seed) + '\\' + 'pyt_save' + '\\' \
                     + 'model' + str(model_iteration) + '.pt'
        self.ac = torch.load(osp.join(osp.dirname(saved_models.__file__), model_path))

        self.index_com = environment_arguments["district_indexes"]
        self.index_part = environment_arguments["building_indexes"]
        self.normalization_value_com = environment_arguments["district_scalars"]
        self.normalization_value_part = environment_arguments["building_scalars"]

        self.len_tot_index = len(self.index_com) + len(self.index_part) * 5

    def set_action_space(self, action_space):
        self.action_space = action_space

    def compute_action(self, observations, building_count):
        """Get observation return action"""

        return self.ppo_policy(observations, building_count)

    def ppo_policy(self, observations, building_count):
        observation_common = [observations[0][i] / n for i, n in zip(self.index_com, self.normalization_value_com)]
        observation_particular = [[o[i] / n for i, n in zip(self.index_part, self.normalization_value_part)] for o in
                                  observations]

        observation_particular = list(itertools.chain(*observation_particular))
        transformed_observation = observation_common + observation_particular

        actions = self.ac.act(torch.as_tensor(transformed_observation, dtype=torch.float32))

        action_list = []

        for a in actions:
            action_list.append(np.array([a], dtype=np.float32))

        return action_list
