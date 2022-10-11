import itertools

from hyperopt import fmin, hp, atpe, tpe, SparkTrials, space_eval, STATUS_OK
import numpy as np
import time
import pyspark
import csv

from tqdm import tqdm

from traineval.evaluation.tune_evaluation import get_specific_action_values, get_observation_weights_search_space, \
    get_observation_weights_search_space_non_ranges

"""
Please do not make changes to this file. 
This is only a reference script provided to allow you 
to do local evaluation. The evaluator **DOES NOT** 
use this script for orchestrating the evaluations. 
"""

from agents.tuning_day_wrapper import OrderEnforcingAgent
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


def evaluate(args, verbose=False):
    if verbose:
        print("Starting local evaluation")

    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingAgent(args)

    obs_dict = env_reset(env)

    agent_time_elapsed = 0

    step_start = time.perf_counter()
    actions = agent.register_reset(obs_dict)
    agent_time_elapsed += time.perf_counter() - step_start

    counter = 1
    day = 1
    avg = 100

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

            # check which day it is for day_tuning
            counter += 1
            if counter % 24 == 0:
                day += 1

            if done:
                episodes_completed += 1
                metrics_t = env.evaluate()
                metrics = {"price_cost": metrics_t[0],
                           "emmision_cost": metrics_t[1],
                           "grid_cost": metrics_t[2]}
                if np.any(np.isnan(metrics_t)):
                    raise ValueError("Episode metrics are nan, please contant organizers")
                episode_metrics.append(metrics)

                if verbose:
                    print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

                obs_dict = env_reset(env)

                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter() - step_start
            else:
                step_start = time.perf_counter()
                actions = agent.compute_action(observations, day)
                agent_time_elapsed += time.perf_counter() - step_start

            num_steps += 1
            if num_steps % 1000 == 0:
                if verbose:
                    print(f"Num Steps: {num_steps}, Num episodes: {episodes_completed}")

            if episodes_completed >= Constants.episodes:
                break
    except KeyboardInterrupt:
        if verbose:
            print("========================= Stopping Evaluation =========================")
        interrupted = True

    if not interrupted:
        if verbose:
            print("=========================Completed=========================")

    if len(episode_metrics) > 0:
        avg_price = np.mean([e['price_cost'] for e in episode_metrics])
        avg_emission = np.mean([e['emmision_cost'] for e in episode_metrics])
        avg_grid = np.mean([e['grid_cost'] for e in episode_metrics])
        avg = np.mean([avg_price, avg_emission, avg_grid])
        if verbose:
            print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
            print("Average Emmision Cost:", np.mean([e['emmision_cost'] for e in episode_metrics]))
            print("Average Grid Cost:", np.mean([e['grid_cost'] for e in episode_metrics]))
            print("Average cost", avg)
            print(f"Total time taken by agent: {agent_time_elapsed}s")

    # if average_cost < Constants.lowest_average_cost:
    #     Constants.lowest_average_cost = average_cost
    #     dict_to_csv([args])

    return {'loss': avg, 'status': STATUS_OK}


def dict_to_csv(dict_list, name):
    observation_values = []

    for key in dict_list[0].keys():
        observation_values.append(key)

    with open(f'tuned_values/optimal_values_{name}.csv', 'w') as csvfile:
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

    ## DAILY WEIGHTED OBSERVATIONS NON-RANGES
    search_space = get_observation_weights_search_space_non_ranges()
    day_params = []
    for day in range(1, 366):  # 13
        search_space["day"] = day
        best_params = fmin(
            fn=evaluate,
            space=search_space,
            algo=tpe.suggest,  # NOTE: You cannot use atpe.suggest with SparkTrials, then use tpe.suggest
            max_evals=50,
            trials=SparkTrials()
        )
        best_params["day"] = day
        day_params.append(best_params)

        print(day)
    dict_to_csv(day_params, "day")

    ### DAILY WEIGHTED OBSERVATIONS ACTIONS
    # search_space = get_specific_action_values()
    # daily_actions = []
    # for day in range(1, 366):
    #     search_space["day"] = day
    #     best_params = fmin(
    #         fn=evaluate,
    #         space=search_space,
    #         algo=tpe.suggest,  # NOTE: You cannot use atpe.suggest with SparkTrials, then use tpe.suggest
    #         max_evals=12,
    #         trials=SparkTrials()
    #     )
    #     best_params["day"] = day
    #     daily_actions.append(best_params)
    # dict_to_csv(daily_actions, "daily_overfit")
    # print(daily_actions)

    ### DAILY WEIGHTED OBSERVATIONS RANGES
    # search_space = get_observation_weights_search_space()
    # day_params = []
    # for day in range(1, 366):  # 13
    #     search_space["day"] = day
    #     best_params = fmin(
    #         fn=evaluate,
    #         space=search_space,
    #         algo=tpe.suggest,  # NOTE: You cannot use atpe.suggest with SparkTrials, then use tpe.suggest
    #         max_evals=50,
    #         trials=SparkTrials()
    #     )
    #     best_params["day"] = day
    #     day_params.append(best_params)
    #
    #     print(day)
    # dict_to_csv(day_params, "day")
