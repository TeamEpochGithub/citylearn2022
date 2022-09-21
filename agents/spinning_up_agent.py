import sys

import numpy as np
import torch
import os.path as osp
import itertools
from train_eval.evaluation import current_model as current_model

class BasicPPOAgent:

    def __init__(self):
        self.action_space = {}
        self.ac = torch.load(osp.join(osp.dirname(current_model.__file__), "model.pt"))

        self.index_com = [0, 2, 19, 4, 8, 24]
        self.index_part = [20, 21, 22, 23]
        self.normalization_value_com = [12, 24, 2, 100, 100, 1]
        self.normalization_value_part = [5, 5, 5, 5]

        self.len_tot_index = len(self.index_com) + len(self.index_part) * 5

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space


    def compute_action(self, observations, building_count):
        """Get observation return action"""

        return self.model_policy(observations, building_count)

    def model_policy(self, observations, building_count):

        observation_common = [observations[0][i] / n for i, n in zip(self.index_com, self.normalization_value_com)]
        observation_particular = [[o[i] / n for i, n in zip(self.index_part, self.normalization_value_part)] for o in observations]

        observation_particular = list(itertools.chain(*observation_particular))
        transformed_observation = observation_common + observation_particular

        raw_actions = self.ac.act(torch.as_tensor(transformed_observation, dtype=torch.float32))

        assert len(raw_actions) == building_count

        action_list = []

        for index, current_raw_action in enumerate(raw_actions):

            #print(index, current_raw_action, self.action_space[index])

            current_action = np.array([current_raw_action], dtype=np.float32)
            #print(self.action_space[index], current_action)
            #assert self.action_space[index].contains(current_action)
            action_list.append(current_action)

        #assert self.action_space[0].contains(action_list)

        return action_list
