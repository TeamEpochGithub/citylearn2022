import csv
from math import sqrt
import random

import pandas as pd
import numpy as np
import os.path as osp


from analysis import analysis_data
from data import citylearn_challenge_2022_phase_1 as competition_data
import torch
import torch.nn as nn

consumptions_path = osp.join(osp.dirname(competition_data.__file__), "consumptions/s_consumptions.csv")

consumptions = pd.read_csv(consumptions_path)[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]


def write_step_to_file(features):
    # action, cost, probability_action, *features
    row = features
    action_file_path = osp.join(osp.dirname(analysis_data.__file__), 'consumption_context.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()


def known_consumption_strategy(loads, solars):
    # Called every hour, after first 24 hours
    # Takes 24 loads and solars
    # Calculates 24 actions
    actions = []
    for ind, load in enumerate(loads):
        actions.append(-(load - solars[ind]) / 6.4)
    # action = -(load-solar) / 6.4
    return actions


def day_night_policy(hour):
    action = -0.067
    if 6 <= hour <= 14:
        action = 0.11
    return action


def calculate_battery_consumption(current_battery_level, prev_battery_level, efficiency):
    loss_coefficient = 0.006  # potentially 0.006?
    battery_consumption = current_battery_level - prev_battery_level * (1 - loss_coefficient)

    if battery_consumption >= 0:
        battery_consumption = battery_consumption / efficiency
    else:
        battery_consumption = battery_consumption * efficiency

    return battery_consumption


def calculate_battery_level(action_capacity, prev_battery_level, efficiency):
    if action_capacity < -5:
        action_capacity = -5
    elif action_capacity > 5:
        action_capacity = 5

    if action_capacity >= 0:
        new_battery_level = min(prev_battery_level + action_capacity * efficiency, 6.4)
    else:
        new_battery_level = max(0, prev_battery_level + action_capacity / efficiency)
    return new_battery_level


# (action *  = 6.4)
def calculate_efficiency(action_capacity):
    if action_capacity < -5:
        action_capacity = -5.0
    elif action_capacity > 5:
        action_capacity = 5.0

    x = np.abs(action_capacity / 5.0)

    if 0 <= x <= 0.3:
        efficiency = sqrt(0.83)
    elif 0.3 < x < 0.7:
        efficiency = sqrt(0.7775 + 0.175 * x)
    elif 0.7 <= x <= 0.8:  # Optimal efficiency
        efficiency = sqrt(0.9)
    else:
        efficiency = sqrt(1.1 - 0.25 * x)
    return efficiency


class ContextualBanditAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.model = ActionNet(input_dim=1, output_dim=1, hidden_shape=(32, 32))
        self.prev_loads = {}
        self.prev_solars = {}
        self.prev_consumptions = {}
        self.prev_battery_spaces = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.prev_loads[agent_id] = []
        self.prev_solars[agent_id] = []
        self.prev_consumptions[agent_id] = []
        self.prev_battery_spaces[agent_id] = []

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        self.timestep += 1
        agent_timestep = self.timestep // len(observation)
        observation = observation[agent_id]
        action = self.action_step(self.action_space[agent_id], observation, agent_timestep, agent_id)
        return np.array([action], dtype=self.action_space[agent_id].dtype)

    def action_step(self, action_space, observation, timestep, agent_id):
        print("timestep: ", timestep)
        if timestep >= 8759:
            return 0

        # self.update_prev_lists(observation, agent_id)

        # if timestep <= 168:
        consumption = consumptions[agent_id][timestep]

        # action = -consumption / 6.4
        # potential_actions = [-0.3, -0.2, -0.15, -0.1, -0.07, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3]
        # action_probability = 1 / len(potential_actions)

        action = 0  # random.choice(potential_actions)
        # loss = self.compute_loss(agent_id, timestep)

        non_shiftable_load = observation[20]
        solar_generation = observation[21]
        electrical_storage_soc = observation[22]
        net_electricity_consumption = observation[23]
        write_step_to_file([non_shiftable_load, solar_generation, electrical_storage_soc])

        # action = self.model.forward(-consumption/6.4)

        return action

    def update_prev_lists(self, observation, agent_id):
        self.prev_loads[agent_id].append(observation[20])
        self.prev_solars[agent_id].append(observation[21])
        self.prev_battery_spaces[agent_id].append(observation[22])
        self.prev_consumptions[agent_id].append(observation[23])

        if len(self.prev_loads) > 25:
            del self.prev_loads[agent_id][0]
            del self.prev_solars[agent_id][0]
            del self.prev_battery_spaces[agent_id][0]

        if len(self.prev_consumptions[agent_id]) > 24:
            del self.prev_consumptions[agent_id][0]

    def compute_loss(self, agent_id, timestep):
        loads = self.prev_loads[agent_id][
                0:24]  # last one excluded since the loads and solars are matched up to the consumptions
        solars = self.prev_solars[agent_id][0:24]

        best_actions = known_consumption_strategy(loads, solars)
        best_consumption = 0
        if timestep < 24:
            prev_battery_level = 0
        else:
            prev_battery_level = self.prev_battery_spaces[agent_id][0]

        for ind, action in enumerate(best_actions):
            action_capacity = action * 6.4
            loc_efficiency = calculate_efficiency(action * 6.4)
            battery_level = calculate_battery_level(action_capacity, prev_battery_level, loc_efficiency)

            battery_consumption = calculate_battery_consumption(battery_level, prev_battery_level, loc_efficiency)

            hour_consumption = loads[ind] + battery_consumption - solars[ind]
            # print(hour_consumption)
            best_consumption += hour_consumption

            prev_battery_level = battery_level
        actual_consumption = sum(self.prev_consumptions[agent_id])
        # print("best consumption: ", best_consumption, " actual consumption: ", actual_consumption)
        return (best_consumption - actual_consumption) ** 2


class ActionNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_shape):
        """
            Policy network. Gives probabilities of picking actions.
        """
        super(ActionNet, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_shape[0]),
            nn.Tanh()
        )

        self.layers = []
        for i, n in enumerate(hidden_shape[:-1]):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(n, hidden_shape[i + 1]),
                    nn.Tanh()
                )
            )

        self.output = torch.nn.Linear(hidden_shape[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)

        return x

    def pick(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.forward(state)
        return action
