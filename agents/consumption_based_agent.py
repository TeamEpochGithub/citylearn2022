import json
import pickle

import numpy as np
from data import citylearn_challenge_2022_phase_1 as competition_data
import os.path as osp


# Should receive all observations
def combined_policy(observation, action_space, consumptions, agent_id, timestep, max_charge):
    num_buildings = len(observation)
    total_consumption = sum([observation[i][23] for i in range(num_buildings)])

    if timestep != len(consumptions) - 1:
        next_consumption = consumptions[timestep + 1]
    else:
        next_consumption = -10

    building_charge = next_consumption / num_buildings / max_charge
    action = -building_charge

    action = np.array([action], dtype=action_space.dtype)

    return action


class ConsumptionBasedAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}
        consumptions_path = osp.join(osp.dirname(competition_data.__file__), "consumptions.pickle")
        self.timestep = -1

        with open(consumptions_path, 'rb') as file:
            self.consumptions = pickle.load(file)

        self.max_charge = 0
        schema_path = osp.join(osp.dirname(competition_data.__file__), "schema.json")
        with open(schema_path) as json_file:
            schema_data = json.load(json_file)
            for k, v in schema_data["buildings"].items():
                charge_val = v["electrical_storage"]["attributes"]["capacity"]
                if charge_val > self.max_charge:
                    self.max_charge = charge_val

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        self.timestep += 1
        return combined_policy(observation, self.action_space[agent_id], self.consumptions, agent_id,
                               self.timestep // 5, self.max_charge)
