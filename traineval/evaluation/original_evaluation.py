import itertools

from hyperopt import fmin, hp, atpe, tpe, SparkTrials, space_eval, STATUS_OK
import numpy as np
import time
import pyspark
import csv

from tqdm import tqdm

"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.orderenforcingwrapper import OrderEnforcingAgent
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


def evaluate(args):
    print("Starting local evaluation")

    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent(args)

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
            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0],
                           "emmision_cost": metrics_t[1],
                           "grid_cost": metrics_t[2]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contant organizers")
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

    # if average_cost < Constants.lowest_average_cost:
    #     Constants.lowest_average_cost = average_cost
    #     dict_to_csv([args])

    return {'loss': average_cost, 'status': STATUS_OK}


def retrieve_search_space():
    search_space = {"price_1": hp.uniform("price_1", -1, 1),
                    "price_2": hp.uniform("price_2", -1, 1),
                    "price_3": hp.uniform("price_3", -1, 1),
                    "price_pred_1": hp.uniform("price_pred_1", -1, 1),
                    "price_pred_2": hp.uniform("price_pred_2", -1, 1),
                    "price_pred_3": hp.uniform("price_pred_3", -1, 1),
                    "carbon_1": hp.uniform("carbon_1", -1, 1),
                    "carbon_2": hp.uniform("carbon_2", -1, 1),
                    "carbon_3": hp.uniform("carbon_3", -1, 1),
                    "solar_1": hp.uniform("solar_1", -1, 1),
                    "solar_2": hp.uniform("solar_2", -1, 1),
                    "solar_3": hp.uniform("solar_3", -1, 1),
                    "solar_diffused_1": hp.uniform("solar_diffused_1", -1, 1),
                    "solar_diffused_2": hp.uniform("solar_diffused_2", -1, 1),
                    "solar_diffused_3": hp.uniform("solar_diffused_3", -1, 1),
                    "solar_direct_1": hp.uniform("solar_direct_1", -1, 1),
                    "solar_direct_2": hp.uniform("solar_direct_2", -1, 1),
                    "solar_direct_3": hp.uniform("solar_direct_3", -1, 1),
                    "hour_1": hp.uniform("hour_1", -1, 1),
                    "hour_2": hp.uniform("hour_2", -1, 1),
                    "hour_3": hp.uniform("hour_3", -1, 1),
                    "storage_1": hp.uniform("storage_1", -1, 1),
                    "storage_2": hp.uniform("storage_2", -1, 1),
                    "storage_3": hp.uniform("storage_3", -1, 1),
                    "consumption_1": hp.uniform("consumption_1", -1, 1),
                    "consumption_2": hp.uniform("consumption_2", -1, 1),
                    "consumption_3": hp.uniform("consumption_3", -1, 1),
                    "load_1": hp.uniform("load_1", -1, 1),
                    "load_2": hp.uniform("load_2", -1, 1),
                    "load_3": hp.uniform("load_3", -1, 1),
                    "temp_1": hp.uniform("temp_1", -1, 1),
                    "temp_2": hp.uniform("temp_2", -1, 1),
                    "temp_3": hp.uniform("temp_3", -1, 1),
                    "humidity_1": hp.uniform("humidity_1", -1, 1),
                    "humidity_2": hp.uniform("humidity_2", -1, 1),
                    "humidity_3": hp.uniform("humidity_3", -1, 1),
                    }
    return search_space


def dict_to_csv(dict_list):
    observation_values = []
    for key in dict_list[0].keys():
        observation_values.append(key)

    with open('optimal_values.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=observation_values)
        writer.writeheader()
        writer.writerows(dict_list)

    print("WRITTEN TO FILE")


if __name__ == '__main__':
    # best_params = fmin(
    #     fn=evaluate,
    #     space=retrieve_search_space(),
    #     algo=tpe.suggest,  # NOTE: You cannot use atpe.suggest with SparkTrials, then use tpe.suggest
    #     max_evals=10,
    #     trials=SparkTrials()
    # )
    # print(best_params)

    search_space = retrieve_search_space()
    month_params = []
    for month in range(1, 13):
        search_space["month"] = month
        best_params = fmin(
            fn=evaluate,
            space=search_space,
            algo=tpe.suggest,  # NOTE: You cannot use atpe.suggest with SparkTrials, then use tpe.suggest
            max_evals=30,
            trials=SparkTrials()
        )
        month_params.append(best_params)

        print(month)
    dict_to_csv(month_params)
