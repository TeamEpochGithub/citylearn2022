import itertools
import sys

import pandas as pd
from hyperopt import fmin, hp, atpe, tpe, SparkTrials, space_eval, STATUS_OK
import numpy as np
import time
#import pyspark
import csv

from tqdm import tqdm

from dynamic_programming.custom_evaluation import evaluate_observation

"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.original_wrapper import OrderEnforcingAgent
from citylearn.citylearn import CityLearnEnv
import os.path as osp
from data import citylearn_challenge_2022_phase_1 as competition_data


class Constants:
    episodes = 1
    schema_path = osp.join(osp.dirname(competition_data.__file__), "schema.json")
    lowest_average_cost = 2


def action_space_to_dict(aspace):
    """ Only for box space """
    return {"high": aspace.high,
            "low": aspace.low,
            "shape": aspace.shape,
            "dtype": str(aspace.dtype)
            }


def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations}
    return obs_dict


def evaluate():
    print("Starting local evaluation")

    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent()

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []

    observation_list = [[], [], [], [], []]
    try:
        while True:

            ### This is only a reference script provided to allow you
            ### to do local evaluation. The evaluator **DOES NOT**
            ### use this script for orchestrating the evaluations.

            observations, _, done, _ = env.step(actions)

            for index, b_obs in enumerate(observations):

                observation_list[index].append(b_obs)

            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0],
                           "emmision_cost": metrics_t[1],
                           "grid_cost": metrics_t[2]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contact organizers")
                episode_metrics.append(metrics)
                print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

                obs_dict = env_reset(env)

                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter() - step_start
            else:


                step_start = time.perf_counter()
                actions = agent.compute_action(observations)
                agent_time_elapsed += time.perf_counter() - step_start

            num_steps += 1
            if num_steps % 1000 == 0:
                print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= Constants.episodes:
                break
    except KeyboardInterrupt:
        print("========================= Stopping Evaluation =========================")
        interrupted = True

    if not interrupted:
        print("=========================Completed=========================")

    if len(episode_metrics) > 0:
        print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
        print("Average Emmision Cost:", np.mean([e['emmision_cost'] for e in episode_metrics]))
        print("Average Grid Cost:", np.mean([e['grid_cost'] for e in episode_metrics]))
        average_cost = np.mean([np.mean([e['price_cost'] for e in episode_metrics]),
                                np.mean([e['emmision_cost'] for e in episode_metrics]),
                                np.mean([e['grid_cost'] for e in episode_metrics])])
        print("Average cost", average_cost)
    print(f"Total time taken by agent: {agent_time_elapsed}s")

    test = ["month", # 0
            "day_type", # 1
            "hour", # 2
            "outdoor_dry_bulb", # 3
            "outdoor_dry_bulb_temperature_predicted_6h", # 4
            "outdoor_dry_bulb_temperature_predicted_12h", # 5
            "outdoor_dry_bulb_temperature_predicted_24h", # 6
            "outdoor_relative_humidity", # 7
            "outdoor_relative_humidity_predicted_6h", # 8
            "outdoor_relative_humidity_predicted_12h", # 9
            "outdoor_relative_humidity_predicted_24h", # 10
            "diffuse_solar_irradiance", # 11
            "diffuse_solar_irradiance_predicted_6h", # 12
            "diffuse_solar_irradiance_predicted_12h", # 13
            "diffuse_solar_irradiance_predicted_24h", # 14
            "direct_solar_irradiance", # 15
            "direct_solar_irradiance_predicted_6h", # 16
            "direct_solar_irradiance_predicted_12h", # 17
            "direct_solar_irradiance_predicted_24h", # 18
            "carbon_intensity", # 19
            "non_shiftable_load", # 20
            "solar_generation", # 21
            "electrical_storage_soc", # 22
            "net_electricity_consumption", # 23
            "electricity_pricing", # 24
            "electricity_pricing_predicted_6h", # 25
            "electricity_pricing_predicted_12h", # 26
            "electricity_pricing_predicted_24h"] # 27

    pd.set_option('display.max_columns', None)

    df_list_solar = []
    df_list_load = []

    for i in range(5):
        observation_df = pd.DataFrame(data=np.asarray(observation_list[i]), columns=test)
        observation_df = observation_df.drop(["net_electricity_consumption"], axis=1)
        observation_df = observation_df.drop(["electrical_storage_soc"], axis=1)
        observation_df["non_shiftable_load_future"] = observation_df["non_shiftable_load"].shift(-1)
        observation_df["solar_generation_future"] = observation_df["solar_generation"].shift(-1)
        observation_df.drop(observation_df.tail(1).index, inplace=True)

        df_list_solar.append(observation_df.drop(["non_shiftable_load", "non_shiftable_load_future"], axis=1))
        df_list_load.append(observation_df.drop(["solar_generation", "solar_generation_future"], axis=1))


    solar = pd.concat(df_list_solar)
    solar.to_csv(r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\consumption_prediction\solar_data.csv", index=False)

    load = pd.concat(df_list_load)
    load.to_csv(r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\consumption_prediction\load_data.csv", index=False)


    sys.exit()

    observation_list = np.asarray(observation_list)

    return evaluate_observation(observation_list)

if __name__ == '__main__':
    print(evaluate())


