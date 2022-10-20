import csv

import numpy as np

from agents.helper_classes.live_learning import LiveLearner
from traineval.training.data_preprocessing import net_electricity_consumption as n
import os.path as osp
from analysis import data_consumption_comparison


def get_chunk_consumptions(agent_id, timestep, consumption_sign, live_learner):
    chunk_consumptions = []
    future_steps = 1

    # while consumptions[agent_id][timestep + future_steps] * consumption_sign > 0:  # Consumptions have the same sign
    while live_learner.predict_multiple_consumptions(future_steps, False)[
        future_steps - 1] * consumption_sign > 0 and future_steps <= 32:
        # next_consumption = consumptions[agent_id][timestep + future_steps]
        next_consumption = live_learner.predict_multiple_consumptions(future_steps)[future_steps - 1]
        chunk_consumptions.append(next_consumption)
        future_steps += 1

        if timestep + future_steps >= 8759:
            break

    return chunk_consumptions


def get_chunk_consumptions_fit_delay(consumption_sign, live_learner):
    max_chunk_size = 32

    chunk_consumptions = live_learner.predict_consumption(max_chunk_size, False)

    for index, consumption in enumerate(chunk_consumptions):

        if consumption * consumption_sign < 0:
            chunk_consumptions = chunk_consumptions[:index]
            break

    return chunk_consumptions


def negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc):
    chunk_total_consumption = sum(chunk_consumptions)

    if -1 * chunk_total_consumption >= (remaining_battery_capacity - soc) / np.sqrt(0.83):
        # If more energy can be obtained than the one necessary to charge the battery
        relative_consumption = [i / chunk_total_consumption for i in chunk_consumptions]
        chunk_charge_loads = [i * (remaining_battery_capacity - soc) / np.sqrt(0.83) for i in relative_consumption]
    else:  # Otherwise charge with all the possible energy
        chunk_charge_loads = [-1 * i for i in chunk_consumptions]

    return chunk_charge_loads


def calculate_next_chunk(consumption_sign, agent_id, timestep, remaining_battery_capacity, soc, live_learner):
    chunk_consumptions = get_chunk_consumptions_fit_delay(consumption_sign, live_learner)
    if len(chunk_consumptions) == 0:
        chunk_consumptions = get_chunk_consumptions_fit_delay(consumption_sign * -1, live_learner)
    if consumption_sign == -1:  # If negative consumption
        chunk_charge_loads = negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc)
    else:
        chunk_charge_loads = chunk_consumptions
    return chunk_charge_loads


def write_step_to_file(agent_id, action, observation):
    # ID, Action, Battery level, Consumption, Load, Solar, Carbon, Price
    row = [agent_id, action, observation[22], observation[23], observation[20], observation[21], observation[19],
           observation[24]]
    action_file_path = osp.join(osp.dirname(data_consumption_comparison.__file__), 'pred_consumption_performance.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()


def pred_consumption_policy(observation, timestep, agent_id, remaining_battery_capacity, soc, live_learner, write_to_file):
    if timestep >= 8759:
        return 0
    # print(timestep, agent_id)

    live_learner.update_lists(observation)

    if timestep < 72:
        hour = observation[2]
        action = -0.067
        if 6 <= hour <= 14:
            action = 0.11
        return action

    next_consumption = live_learner.predict_consumption(1, True)[0]

    if next_consumption == 0:
        return 0
    elif next_consumption > 0:
        consumption_sign = 1
    else:
        consumption_sign = -1

    chunk_charge_loads = calculate_next_chunk(consumption_sign, agent_id, timestep, remaining_battery_capacity, soc,
                                              live_learner)
    charge_load = -1 * consumption_sign * chunk_charge_loads[0]
    action = charge_load / remaining_battery_capacity

    if write_to_file:
        write_step_to_file(agent_id, action, observation)

    return action


class TimeStepPredConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.remaining_battery_capacity = {}
        self.soc = {}

        self.live_learners = {}
        self.write_to_file = True

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.remaining_battery_capacity[agent_id] = 6.4
        self.soc[agent_id] = 0

        if str(agent_id) not in self.live_learners:
            self.live_learners[str(agent_id)] = LiveLearner(800, 15, self.write_to_file)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        building_timestep = self.timestep // len(observation)
        observation = observation[agent_id]

        action_out = pred_consumption_policy(observation, building_timestep, agent_id,
                                             self.remaining_battery_capacity[agent_id],
                                             self.soc[agent_id], self.live_learners[str(agent_id)], self.write_to_file)

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
