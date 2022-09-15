import sys

import numpy as np
import pandas as pd


def simple_policy(observation, action_space):
    """
    Simple rule based policy based on day or night time
    """

    carbon_intensity = observation[19]
    electricity_pricing = observation[24]
    score = (carbon_intensity + electricity_pricing)*-1

    # print(score)
    # print("STORAGE", electrical_storage_soc)
    # print("CONSUMPTION", net_electricity_consumption)

    if score > -0.4:
        action = 0.05
    elif score > -0.5:
        action = 0.05
    elif score > -0.6:
        action = -0.1
    elif score > -0.7:
        action = -0.15
    else:
        action = -0.2

    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action


class BasicRewardAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return simple_policy(observation, self.action_space[agent_id])
