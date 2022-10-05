import sys

import numpy as np

from traineval.utils.convert_arguments import get_max_scalars, get_min_scalars


# def optimized_combined_policy(observation, action_space, args):


def scale_observation_vals(observation):
    maxs = get_max_scalars()
    mins = get_min_scalars()

    scaled_observation = []
    for ind, v in enumerate(observation):
        maxim = maxs[ind]
        minim = mins[ind]
        scaled_observation.append((((v - minim) * (1 - (-1))) / (minim - minim)) + (-1))

    return scaled_observation


def combined_policy(observation, action_space, args, day):
    observation = scale_observation_vals(observation)
    month = observation[0]
    day_type = observation[1]
    hour = observation[2]
    outdoor_dry_bulb_temperature = observation[3]
    outdoor_dry_bulb_temperature_predicted_6h = observation[4]
    outdoor_dry_bulb_temperature_predicted_12h = observation[5]
    outdoor_dry_bulb_temperature_predicted_24h = observation[6]
    outdoor_relative_humidity = observation[7]
    outdoor_relative_humidity_predicted_6h = observation[8]
    outdoor_relative_humidity_predicted_12h = observation[9]
    outdoor_relative_humidity_predicted_24h = observation[10]
    diffuse_solar_irradiance = observation[11]
    diffuse_solar_irradiance_predicted_6h = observation[12]
    diffuse_solar_irradiance_predicted_12h = observation[13]
    diffuse_solar_irradiance_predicted_24h = observation[14]
    direct_solar_irradiance = observation[15]
    direct_solar_irradiance_predicted_6h = observation[16]
    direct_solar_irradiance_predicted_12h = observation[17]
    direct_solar_irradiance_predicted_24h = observation[18]
    carbon_intensity = observation[19]
    non_shiftable_load = observation[20]
    solar_generation = observation[21]
    electrical_storage_soc = observation[22]
    net_electricity_consumption = observation[23]
    electricity_pricing = observation[24]
    electricity_pricing_predicted_6h = observation[25]
    electricity_pricing_predicted_12h = observation[26]
    electricity_pricing_predicted_24h = observation[27]

    if day != args["day"]:
        return np.array([0], dtype=action_space.dtype)

    ### PRICE
    pricing_action = args["price_1"] * electricity_pricing
    pricing_pred_action = args["price_pred_1"] * electricity_pricing_predicted_6h

    ### EMISSION
    carbon_action = args["carbon_1"] * carbon_intensity
    generation_action = args["solar_1"] * solar_generation
    diffuse_action = args["solar_diffused_1"] * diffuse_solar_irradiance
    direct_action = args["solar_direct_1"] * direct_solar_irradiance

    ### GRID
    hour_action = 1
    if 6 < hour <= 14:
        hour_action *= args["hour_1"]
    elif 14 < hour <= 23:
        hour_action *= args["hour_2"]
    else:
        hour_action *= args["hour_3"]
    storage_action = args["storage_1"] * electrical_storage_soc
    consumption_action = args["consumption_1"] * net_electricity_consumption
    load_action = args["load_1"] * non_shiftable_load
    temp_action = args["temp_1"] * outdoor_dry_bulb_temperature
    humidity_action = args["humidity_1"] * outdoor_relative_humidity

    price_action = np.average([pricing_action, pricing_pred_action])
    emission_action = np.average([carbon_action, generation_action, diffuse_action, direct_action])
    grid_action = np.average(
        [hour_action, storage_action, consumption_action, load_action, temp_action, humidity_action])

    action_average = (price_action + emission_action + grid_action) / 3

    action = np.array([action_average], dtype=action_space.dtype)

    return action


# Might be a lot faster than doing it range-based
class TunableDayNoRangesAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self, args):
        self.action_space = {}
        self.args = args

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id, day):
        """Get observation return action"""
        return combined_policy(observation, self.action_space[agent_id], self.args, day)
