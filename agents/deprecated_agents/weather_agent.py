import sys

import numpy as np
import pandas as pd


def weather_policy(observation, action_space):
    """
    Simple rule based policy based on day or night time
    """
    hour = observation[2]  # Hour index is 2 for all observations

    diffuse_solar_irradiance = observation[11]
    direct_solar_irradiance = observation[15]
    global_solar_irradiance = diffuse_solar_irradiance + direct_solar_irradiance

    sixhour_global_solar_irradiance = observation[12] + observation[16]
    twelvehour_global_solar_irradiance = observation[13] + observation[17]
    twentyfourhour_global_solar_irradiance = observation[14] + observation[18]

    carbon_intensity = observation[19]
    # print(carbon_intensity)

    electrical_storage_soc = observation[22]
    net_electricity_consumption = observation[23]
    electricity_pricing = observation[24]


    # print((carbon_intensity + electricity_pricing)*-1)

    print("STORAGE", electrical_storage_soc)
    print("CONSUMPTION", net_electricity_consumption)

    action_vals = [-0.3, -0.2, -0.15, -0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1, 0.2]

    if global_solar_irradiance < sixhour_global_solar_irradiance:
        action = 0.15
    else:
        action = -0.1

    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action


class BasicWeatherAgent:
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
        return weather_policy(observation, self.action_space[agent_id])
