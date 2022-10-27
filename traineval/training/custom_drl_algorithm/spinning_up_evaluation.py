import os.path as osp
import time

import numpy as np
from citylearn.citylearn import CityLearnEnv

# from agents.order_enforcing_wrapper_spinning_up import OrderEnforcingSpinningUpAgent
from agents.deprecated_agents.order_enforcing_wrapper_spinning_up import OrderEnforcingSpinningUpAgent
from data import citylearn_challenge_2022_phase_1 as competition_data
from traineval.utils.convert_arguments import get_environment_arguments


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


def evaluate(environment_arguments, model_type, model_seed, model_iteration, verbose=True):
    if verbose:
        print("Starting local evaluation")

    env = CityLearnEnv(schema=Constants.schema_path)
    agent = OrderEnforcingSpinningUpAgent(environment_arguments, model_type, model_seed, model_iteration)

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
                    raise ValueError("Episode metrics are nan, please contact organizers")
                episode_metrics.append(metrics)
                if verbose:
                    print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

                obs_dict = env_reset(env)

                step_start = time.perf_counter()
                actions = agent.register_reset(obs_dict)
                agent_time_elapsed += time.perf_counter() - step_start
            else:
                step_start = time.perf_counter()
                actions = agent.compute_action(observations)
                # print("acts: ", actions, "obs: ", observations)
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
        if verbose:
            print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
            print("Average Emmision Cost:", np.mean([e['emmision_cost'] for e in episode_metrics]))
            print("Average Grid Cost:", np.mean([e['grid_cost'] for e in episode_metrics]))
        average_cost = np.mean([np.mean([e['price_cost'] for e in episode_metrics]),
                                np.mean([e['emmision_cost'] for e in episode_metrics]),
                                np.mean([e['grid_cost'] for e in episode_metrics])])
        if verbose:
            print("Average cost:", average_cost)
        return average_cost, agent_time_elapsed
    if verbose:
        print(f"Total time taken by agent: {agent_time_elapsed}s")


if __name__ == '__main__':
    # district_args = ["hour", "carbon_intensity", "electricity_pricing"]
    # building_args = ["electrical_storage_soc"]

    district_args = ["hour",
                     "month",
                     "carbon_intensity",
                     "electricity_pricing"]

    building_args = ["non_shiftable_load",
                     "solar_generation",
                     "electrical_storage_soc",
                     "net_electricity_consumption"]

    environment_arguments = get_environment_arguments(district_args, building_args)

    evaluate(environment_arguments, "ppo", 0, 0, True)

