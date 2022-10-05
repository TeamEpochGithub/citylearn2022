import sys

import numpy as np
import csv
import os.path as osp
import traineval.evaluation.tuned_values as value_dir


def combined_policy(observation, action_space):
    values_path = osp.join(osp.dirname(value_dir.__file__), "optimal_values_month.csv")
    all_args = list(csv.DictReader(open(values_path), quoting=csv.QUOTE_ALL))
    # lst = list(csv.DictReader(open('../traineval/evaluation/tuned_values/optimal_values_month.csv')));
    # args = lst[0]
    month = observation[0]
    args = None
    for arg_dict in all_args:
        if int(arg_dict["month"]) == int(month):
            args = arg_dict
            break
    # args = all_args[int(month) - 1]
    for k, v in args.items():
        args[k] = float(v)

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

    ### PRICE
    pricing_action = 1
    if 0 < electricity_pricing <= 0.21:
        pricing_action *= args["price_1"]
    elif 0.21 < electricity_pricing <= 0.22:
        pricing_action *= args["price_2"]
    else:
        pricing_action *= args["price_3"]

    pricing_pred_action = 1
    if 0 < electricity_pricing_predicted_6h <= 0.21:
        pricing_pred_action *= args["price_pred_1"]
    elif 0.21 < electricity_pricing_predicted_6h <= 0.22:
        pricing_pred_action *= args["price_pred_2"]
    else:
        pricing_pred_action *= args["price_pred_3"]

    ### EMISSION
    carbon_action = 1
    if 0 < carbon_intensity <= 0.139231:
        carbon_action *= args["carbon_1"]
    elif 0.139231 < carbon_intensity <= 0.169461:
        carbon_action *= args["carbon_2"]
    else:
        carbon_action *= args["carbon_3"]

    generation_action = 1
    if 0 < solar_generation <= 0:
        generation_action *= args["solar_1"]
    elif 0 < solar_generation <= 163.14452:
        generation_action *= args["solar_2"]
    else:
        generation_action *= args["solar_3"]

    diffuse_action = 1
    if 0 < diffuse_solar_irradiance <= 0:
        diffuse_action *= args["solar_diffused_1"]
    elif 0 < diffuse_solar_irradiance <= 216:
        diffuse_action *= args["solar_diffused_2"]
    else:
        diffuse_action *= args["solar_diffused_3"]

    direct_action = 1
    if 0 < direct_solar_irradiance <= 0:
        direct_action *= args["solar_direct_1"]
    elif 0 < direct_solar_irradiance <= 141:
        direct_action *= args["solar_direct_2"]
    else:
        direct_action *= args["solar_direct_3"]

    ### GRID
    hour_action = 1
    if 6 < hour <= 14:
        hour_action *= args["hour_1"]
    elif 14 < hour <= 23:
        hour_action *= args["hour_2"]
    else:
        hour_action *= args["hour_3"]

    storage_action = 1
    if 0 < electrical_storage_soc <= 0.33:
        storage_action *= args["storage_1"]
    elif 0.33 < electrical_storage_soc <= 0.66:
        storage_action *= args["storage_2"]
    else:
        storage_action *= args["storage_3"]

    consumption_action = 1
    if 0 < net_electricity_consumption <= 0.6:
        consumption_action *= args["consumption_1"]
    elif 0.6 < net_electricity_consumption <= 1.2:
        consumption_action *= args["consumption_2"]
    else:
        consumption_action *= args["consumption_3"]

    load_action = 1
    if 0 < non_shiftable_load <= 0.726493:
        load_action *= args["load_1"]
    elif 0.726493 < non_shiftable_load <= 1.185376:
        load_action *= args["load_2"]
    else:
        load_action *= args["load_3"]

    temp_action = 1
    if 0 < outdoor_dry_bulb_temperature <= 15.6:
        temp_action *= args["temp_1"]
    elif 15.6 < outdoor_dry_bulb_temperature <= 18.3:
        temp_action *= args["temp_2"]
    else:
        temp_action *= args["temp_3"]

    humidity_action = 1
    if 0 < outdoor_relative_humidity <= 69:
        humidity_action *= args["humidity_1"]
    elif 69 < outdoor_relative_humidity <= 81:
        humidity_action *= args["humidity_2"]
    else:
        humidity_action *= args["humidity_3"]

    price_action = np.average([pricing_action, pricing_pred_action])
    emission_action = np.average([carbon_action, generation_action, diffuse_action, direct_action])
    grid_action = np.average(
        [hour_action, storage_action, consumption_action, load_action, temp_action, humidity_action])

    action_average = (price_action + emission_action + grid_action) / 3

    action = np.array([action_average], dtype=action_space.dtype)

    return action


class MonthTunedAgent:
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
        return combined_policy(observation, self.action_space[agent_id])
