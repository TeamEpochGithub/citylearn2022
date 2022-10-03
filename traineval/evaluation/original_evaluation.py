import itertools

from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter
from smac.facade.smac_bb_facade import SMAC4BB
from smac.scenario.scenario import Scenario
import numpy as np
import time

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
    return average_cost


if __name__ == '__main__':

    args = None

    # Define your hyperparameters
    configspace = ConfigurationSpace()
    configspace.add_hyperparameter(UniformFloatHyperparameter("price_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("price_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("price_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("price_pred_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("price_pred_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("price_pred_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("carbon_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("carbon_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("carbon_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("solar_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("solar_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("solar_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("solar_diffused_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("solar_diffused_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("solar_diffused_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("solar_direct_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("solar_direct_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("solar_direct_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("hour_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("hour_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("hour_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("storage_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("storage_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("storage_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("consumption_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("consumption_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("consumption_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("load_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("load_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("load_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("temp_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("temp_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("temp_3", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("humidity_1", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("humidity_2", -1, 1))
    configspace.add_hyperparameter(UniformFloatHyperparameter("humidity_3", -1, 1))

    # Provide meta data for the optimization
    scenario = Scenario({
        "run_obj": "quality",  # Optimize quality (alternatively runtime)
        "runcount-limit": 100,  # Max number of function evaluations (the more the better)
        "cs": configspace,
    })

    smac = SMAC4BB(scenario=scenario, tae_runner=evaluate)
    best_found_config = smac.optimize()

    evaluate(args)
