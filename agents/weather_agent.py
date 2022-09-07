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

    # cooling_storage_soc = observation[25]
    # heating_storage_soc = observation[26]
    # dhw_storage_soc = observation[27]
    electrical_storage_soc = observation[22]
    net_electricity_consumption = observation[23]

    print("STORAGE", electrical_storage_soc)
    # print("CONSUMPTION", net_electricity_consumption)

    if global_solar_irradiance < sixhour_global_solar_irradiance:
        action = 0.15
    else:
        action = -0.1

    # action = 0.0
    # if 1 <= hour <= 6:
    #     action = 0.05532
    #
    # elif 7 <= hour <= 15:
    #     action = -0.02
    #
    # elif 16 <= hour <= 18:
    #     action = -0.0044
    #
    # elif 19 <= hour <= 22:
    #     action = -0.024
    #
    # elif 23 <= hour <= 24:
    #     action = 0.034
    #
    # else:
    #     action = 0.0

    # print(((0.05532 * 6) + (0.034 * 1)))
    # print(((-0.02 * 8) + (-0.0044 * 4) + (-0.024 * 3)))

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
