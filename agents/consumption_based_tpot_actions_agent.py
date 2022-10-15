import csv
import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import numpy as np
from data import citylearn_challenge_2022_phase_1 as competition_data
import os.path as osp
import joblib
import traineval.training.tpot_actions as tpot_files


# Should receive all observations
from traineval.utils.convert_arguments import environment_convert_argument


def combined_policy(observation, action_space, agent_id, action_model):
    arguments = environment_convert_argument(["month",
                                              "day_type",
                                              "hour",
                                              "outdoor_dry_bulb",
                                              "outdoor_relative_humidity",
                                              "diffuse_solar_irradiance",
                                              "direct_solar_irradiance",
                                              "carbon_intensity",
                                              "non_shiftable_load",
                                              "solar_generation",
                                              "electrical_storage_soc",
                                              "net_electricity_consumption",
                                              "electricity_pricing"])
    obs = []
    for arg in arguments:
        obs.append(observation[agent_id][arg])

    action = action_model.predict([obs])
    action = np.array([action], dtype=action_space.dtype)

    return action


class ConsumptionBasedTPOTActionsAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.max_charge = 6.4

        tpot_model_path = osp.join(osp.dirname(tpot_files.__file__), 'pipe.joblib')
        self.action_model = joblib.load(tpot_model_path)

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        self.timestep += 1
        return combined_policy(observation, self.action_space[agent_id], agent_id, self.action_model)
