import sys

import numpy as np


def combined_policy(observation, action_space, args):
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

    price_action = 1
    if 0 < electricity_pricing <= 0.21:
        price_action *= args["price_1"]
    elif 0.21 < electricity_pricing <= 0.22:
        price_action *= args["price_2"]
    else:
        price_action *= args["price_3"]

    if 0 < electricity_pricing_predicted_6h <= 0.21:
        price_action *= args["price_pred_1"]
    elif 0.21 < electricity_pricing_predicted_6h <= 0.22:
        price_action *= args["price_pred_2"]
    else:
        price_action *= args["price_pred_3"]

    emission_action = 1
    if 0 < carbon_intensity <= 0.139231:
        emission_action *= args["carbon_1"]
    elif 0.139231 < carbon_intensity <= 0.169461:
        emission_action *= args["carbon_2"]
    else:
        emission_action *= args["carbon_3"]

    if 0 < solar_generation <= 0:
        emission_action *= args["solar_1"]
    elif 0 < solar_generation <= 163.14452:
        emission_action *= args["solar_2"]
    else:
        emission_action *= args["solar_3"]

    if 0 < diffuse_solar_irradiance <= 0:
        emission_action *= args["solar_diffused_1"]
    elif 0 < diffuse_solar_irradiance <= 216:
        emission_action *= args["solar_diffused_2"]
    else:
        emission_action *= args["solar_diffused_3"]

    if 0 < direct_solar_irradiance <= 0:
        emission_action *= args["solar_direct_1"]
    elif 0 < direct_solar_irradiance <= 141:
        emission_action *= args["solar_direct_2"]
    else:
        emission_action *= args["solar_direct_3"]

    grid_action = 1

    if 6 < hour <= 14:
        grid_action *= args["hour_1"]
    elif 14 < hour <= 23:
        grid_action *= args["hour_2"]
    else:
        grid_action *= args["hour_3"]

    if 0 < electrical_storage_soc <= 0.33:
        grid_action *= args["storage_1"]
    elif 0.33 < electrical_storage_soc <= 0.66:
        grid_action *= args["storage_2"]
    else:
        grid_action *= args["storage_3"]

    if 0 < net_electricity_consumption <= 0.6:
        grid_action *= args["consumption_1"]
    elif 0.6 < net_electricity_consumption <= 1.2:
        grid_action *= args["consumption_2"]
    else:
        grid_action *= args["consumption_3"]

    if 0 < non_shiftable_load <= 0.726493:
        grid_action *= args["load_1"]
    elif 0.726493 < non_shiftable_load <= 1.185376:
        grid_action *= args["load_2"]
    else:
        grid_action *= args["load_3"]

    if 0 < outdoor_dry_bulb_temperature <= 15.6:
        grid_action *= args["temp_1"]
    elif 15.6 < outdoor_dry_bulb_temperature <= 18.3:
        grid_action *= args["temp_2"]
    else:
        grid_action *= args["temp_3"]

    if 0 < outdoor_relative_humidity <= 69:
        grid_action *= args["humidity_1"]
    elif 69 < outdoor_relative_humidity <= 81:
        grid_action *= args["humidity_2"]
    else:
        grid_action *= args["humidity_3"]

    return (price_action + emission_action + grid_action) / 3


class MultiPolicyAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        action_space = {}

    def set_action_space(self, agent_id, action_space):
        action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return combined_policy(observation, action_space[agent_id])