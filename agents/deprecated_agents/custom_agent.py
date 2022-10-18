import itertools
import os.path as osp

import joblib
import numpy as np
import torch
import pickle

import agents.spinning_agent as agentfile


class CustomDRLAgent:

    def __init__(self, environment_arguments, model_type: str, model_seed: int, model_iteration: int):
        self.action_space = {}

        # model_path = 'agents/trained_models/wowamodela.pt'
        # print(osp.join(osp.dirname(trained_models.__file__), wowamodela.pt))
        # self.ac = torch.load(osp.join(osp.dirname(saved_models.__file__), model_path))

        # self.ac = torch.load(osp.join(osp.dirname(trained_models.__file__), "wowamodela.pt"))
        self.ac = torch.load(osp.join(osp.dirname(agentfile.__file__),
                                      "../traineval/training/custom_drl_algorithm/wowamodela.pt"))

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

        actions = self.ac.pick(np.array(transformed_observation))

        action_list = []

        return [actions.tolist()[0]] * building_count
