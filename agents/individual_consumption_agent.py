import sys
import pandas as pd
import numpy as np
import os.path as osp
from data import citylearn_challenge_2022_phase_1 as competition_data

consumptions_path = osp.join(osp.dirname(competition_data.__file__), "consumptions/s_consumptions.csv")

consumptions = pd.read_csv(consumptions_path)[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]


def individual_consumption_policy(action_space, time_step, agent_id):
    if time_step >= 8759:
        return np.array([0], dtype=action_space.dtype)

    consumption = consumptions[agent_id][time_step]

    action = -consumption / 6.4

    action = np.array([action], dtype=action_space.dtype)
    return action


class IndividualConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        self.timestep += 1
        collaborative_timestep = self.timestep // 5
        return individual_consumption_policy(self.action_space[agent_id], collaborative_timestep, agent_id)
