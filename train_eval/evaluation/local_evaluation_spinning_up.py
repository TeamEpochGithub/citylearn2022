import itertools
import sys

import numpy as np
import time
import pandas as pd

"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.order_enforcing_wrapper_spinning_up import OrderEnforcingSpinningUpAgent
from citylearn.citylearn import CityLearnEnv
import os.path as osp
from data import citylearn_challenge_2022_phase_1 as competition_data


class Constants:
    episodes = 1
    schema_path = osp.join(osp.dirname(competition_data.__file__), "schema.json")

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


def evaluate(current_agent):
    print("Starting local evaluation")

    obs_list = []

    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingSpinningUpAgent(current_agent)

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    episodes_completed = 0
    num_steps = 0
    interrupted = False
    episode_metrics = []
    try:
        while True:

            ### This is only a reference script provided to allow you 
            ### to do local evaluation. The evaluator **DOES NOT** 
            ### use this script for orchestrating the evaluations.

            observations, _, done, _ = env.step(actions)

            obs_list.append(observations[0] + observations[1][-8:] + observations[2][-8:] + observations[3][-8:] + observations[4][-8:])

            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0], "emmision_cost": metrics_t[1]}
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
        print("Average Total Cost:", (
                np.mean([e['emmision_cost'] for e in episode_metrics] + np.mean(
                    [e['price_cost'] for e in episode_metrics])) / 2))
    print(f"Total time taken by agent: {agent_time_elapsed}s")

    df = pd.DataFrame(obs_list, columns= ["month",
                            "day_type",
                            "hour",
                            "outdoor_dry_bulb",
                            "outdoor_dry_bulb_temperature_predicted_6h",
                            "outdoor_dry_bulb_temperature_predicted_12h",
                            "outdoor_dry_bulb_temperature_predicted_24h",
                            "outdoor_relative_humidity",
                            "outdoor_relative_humidity_predicted_6h",
                            "outdoor_relative_humidity_predicted_12h",
                            "outdoor_relative_humidity_predicted_24h",
                            "diffuse_solar_irradiance",
                            "diffuse_solar_irradiance_predicted_6h",
                            "diffuse_solar_irradiance_predicted_12h",
                            "diffuse_solar_irradiance_predicted_24h",
                            "direct_solar_irradiance",
                            "direct_solar_irradiance_predicted_6h",
                            "direct_solar_irradiance_predicted_12h",
                            "direct_solar_irradiance_predicted_24h",
                            "carbon_intensity",
                            "non_shiftable_load",
                            "solar_generation",
                            "electrical_storage_soc",
                            "net_electricity_consumption",
                            "electricity_pricing",
                            "electricity_pricing_predicted_6h",
                            "electricity_pricing_predicted_12h",
                            "electricity_pricing_predicted_24h",
                                          "non_shiftable_load",
                                          "solar_generation",
                                          "electrical_storage_soc",
                                          "net_electricity_consumption",
                                          "electricity_pricing",
                                          "electricity_pricing_predicted_6h",
                                          "electricity_pricing_predicted_12h",
                                          "electricity_pricing_predicted_24h",
                                          "non_shiftable_load",
                                          "solar_generation",
                                          "electrical_storage_soc",
                                          "net_electricity_consumption",
                                          "electricity_pricing",
                                          "electricity_pricing_predicted_6h",
                                          "electricity_pricing_predicted_12h",
                                          "electricity_pricing_predicted_24h",
                                          "non_shiftable_load",
                                          "solar_generation",
                                          "electrical_storage_soc",
                                          "net_electricity_consumption",
                                          "electricity_pricing",
                                          "electricity_pricing_predicted_6h",
                                          "electricity_pricing_predicted_12h",
                                          "electricity_pricing_predicted_24h",
                                          "non_shiftable_load",
                                          "solar_generation",
                                          "electrical_storage_soc",
                                          "net_electricity_consumption",
                                          "electricity_pricing",
                                          "electricity_pricing_predicted_6h",
                                          "electricity_pricing_predicted_12h",
                                          "electricity_pricing_predicted_24h"
                                          ])

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    print(df.describe())


if __name__ == '__main__':
    from agents.rbc_agent import BasicRBCAgent
    from agents.spinning_up_agent import BasicPPOAgent
    evaluate(BasicPPOAgent)
