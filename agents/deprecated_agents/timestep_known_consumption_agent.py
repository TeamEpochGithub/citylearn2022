import csv
import sys
import pandas as pd
import numpy as np
import os.path as osp

from analysis import analysis_data
from data import citylearn_challenge_2022_phase_1 as competition_data

from traineval.training.data_preprocessing import net_electricity_consumption as n
from traineval.training.data_preprocessing.pricing_simplified import pricing, shift_date

consumptions_path = osp.join(osp.dirname(competition_data.__file__), "consumptions/building_consumptions.csv")
# carbon_path = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")

consumptions = pd.read_csv(consumptions_path)[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]


# carbon = pd.read_csv(carbon_path)["kg_CO2/kWh"]
# carbon = carbon.values.tolist()[1:]

def get_chunk_consumptions(agent_id, timestep, consumption_sign):
    chunk_consumptions = []
    future_steps = 0

    while consumptions[agent_id][timestep + future_steps] * consumption_sign > 0:  # Consumptions have the same sign
        next_consumption = consumptions[agent_id][timestep + future_steps]
        chunk_consumptions.append(next_consumption)
        future_steps += 1

        if timestep + future_steps >= 8759:
            break

    return chunk_consumptions


def negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc):
    chunk_total_consumption = sum(chunk_consumptions)

    if -1 * chunk_total_consumption >= (remaining_battery_capacity - soc) / np.sqrt(0.83):
        # If more energy can be obtained than the one necessary to charge the battery
        relative_consumption = [i / chunk_total_consumption for i in chunk_consumptions]
        energies = [i * (remaining_battery_capacity - soc) / np.sqrt(0.83) for i in relative_consumption]
    else:  # Otherwise charge with all the possible energy
        energies = [-1 * i for i in chunk_consumptions]

    return energies


def calculate_next_chunk(consumption_sign, agent_id, timestep, remaining_battery_capacity, soc):
    chunk_consumptions = get_chunk_consumptions(agent_id, timestep, consumption_sign)
    if consumption_sign == -1:  # If negative consumption
        energies = negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc)
    else:
        energies = chunk_consumptions
    return energies


def write_step_to_file(agent_id, action, observation):
    # ID, Action, Battery level, Consumption, Load, Solar, Carbon, Price
    row = [agent_id, action, observation[22], observation[23], observation[20], observation[21], observation[19],
           observation[24]]
    action_file_path = osp.join(osp.dirname(analysis_data.__file__), 'known_consumption_performance.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()


def write_historic_consumptions_to_file(agent_id, timestep):

    num_consumptions = 10
    row = []
    for i in range(num_consumptions):
        if timestep - i - 2 >= 0:
            row.append(consumptions[agent_id][timestep-i - 2])
        else:
            row.append(0)

    # row = [agent_id, ]
    action_file_path = osp.join(osp.dirname(analysis_data.__file__), 'historic_consumptions.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()


def individual_consumption_policy(observation, timestep, agent_id, remaining_battery_capacity, soc, write_to_file):
    if timestep >= 8759:
        return 0

    next_consumption = consumptions[agent_id][timestep]

    if next_consumption == 0:
        return 0
    elif next_consumption > 0:
        consumption_sign = 1
    else:
        consumption_sign = -1

    energies = calculate_next_chunk(consumption_sign, agent_id, timestep, remaining_battery_capacity, soc)
    energy = -1 * consumption_sign * energies[0]

    action = energy / remaining_battery_capacity

    if write_to_file:
        write_step_to_file(agent_id, action, observation)
        write_historic_consumptions_to_file(agent_id, timestep)

    return action


class TimeStepKnownConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.remaining_battery_capacity = {}
        self.soc = {}
        self.write_to_file = True

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.remaining_battery_capacity[agent_id] = 6.4
        self.soc[agent_id] = 0

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        building_timestep = self.timestep // len(observation)
        observation = observation[agent_id]

        action_out = individual_consumption_policy(observation, building_timestep, agent_id,
                                                   self.remaining_battery_capacity[agent_id],
                                                   self.soc[agent_id], self.write_to_file)

        action = float(np.array(action_out, dtype=self.action_space[agent_id].dtype))
        max_power = n.max_power(self.soc[agent_id], 5, self.remaining_battery_capacity[agent_id])
        energy = n.energy_normed(action * self.remaining_battery_capacity[agent_id], max_power)
        efficiency = n.efficiency(energy, 5)

        previous_soc = self.soc[agent_id]
        self.soc[agent_id] = n.soc(energy, previous_soc, efficiency, self.remaining_battery_capacity[agent_id])

        battery_cons = n.last_energy_balance(self.soc[agent_id], previous_soc, efficiency)
        self.remaining_battery_capacity[agent_id] = n.new_capacity(self.remaining_battery_capacity[agent_id],
                                                                   battery_cons)

        return np.array([action_out], dtype=self.action_space[agent_id].dtype)