import sys
import pandas as pd
import numpy as np
import os.path as osp

from agents.helper_classes.live_learning import LiveLearner
from data import citylearn_challenge_2022_phase_1 as competition_data

from traineval.training.data_preprocessing import net_electricity_consumption as n
from traineval.training.data_preprocessing.pricing_simplified import pricing, shift_date

consumptions_path = osp.join(osp.dirname(competition_data.__file__), "consumptions/building_consumptions.csv")
# carbon_path = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")

consumptions = pd.read_csv(consumptions_path)[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]


# carbon = pd.read_csv(carbon_path)["kg_CO2/kWh"]
# carbon = carbon.values.tolist()[1:]

def get_chunk_consumptions(agent_id, timestep, consumption_sign, live_learner):
    chunk_consumptions = []
    future_steps = 1

    # while consumptions[agent_id][timestep + future_steps] * consumption_sign > 0:  # Consumptions have the same sign
    while live_learner.predict_multiple_consumptions(future_steps)[future_steps - 1] * consumption_sign > 0 and future_steps <= 12:
        # print(live_learner.predict_multiple_consumptions(future_steps)[future_steps - 1])
        next_consumption = live_learner.predict_multiple_consumptions(future_steps)[future_steps - 1]
        # next_consumption = consumptions[agent_id][timestep + future_steps]
        chunk_consumptions.append(next_consumption)
        future_steps += 1
        if timestep + future_steps >= 8759:
            break

    return chunk_consumptions

def get_chunk_consumptions_fit_delay(consumption_sign, live_learner):

    live_learner.force_fit()

    max_chunk_size = 12

    chunk_consumptions = live_learner.fit_delay_buffer_consumption(max_chunk_size)

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


def calculate_next_chunk(prev_consumption_sign, agent_id, timestep, remaining_battery_capacity, soc, live_learner):
    consumption_sign = -prev_consumption_sign
    chunk_consumptions = get_chunk_consumptions_fit_delay(consumption_sign, live_learner)
    if consumption_sign == -1:  # If negative consumption
        chunk_charge_loads = negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc)
    else:
        chunk_charge_loads = chunk_consumptions

    if not chunk_charge_loads:
        chunk_charge_loads = calculate_next_chunk(consumption_sign, agent_id, timestep, remaining_battery_capacity, soc, live_learner)

    return chunk_charge_loads


def pred_consumption_policy(observation, timestep, agent_id, remaining_battery_capacity, soc,
                            prev_consumption_sign, chunk_charge_loads, step_in_chunk, live_learner, prev_predicted_chunk_size):
    if timestep >= 8759:
        return 0, chunk_charge_loads, step_in_chunk, prev_consumption_sign, prev_predicted_chunk_size

    live_learner.update_lists(observation)

    if timestep < 150:  # Can be lowered to just above the largest lag value when the agent is operational.
        hour = observation[2]
        action = -0.067
        if 6 <= hour <= 14:
            action = 0.11
        return action, chunk_charge_loads, step_in_chunk + 1, prev_consumption_sign, prev_predicted_chunk_size

    # next_consumption = consumptions[agent_id][timestep]
    next_consumption = live_learner.fit_delay_buffer_consumption(1)[0]

    if step_in_chunk >= prev_predicted_chunk_size - 1 or next_consumption * prev_consumption_sign < 0:
        # This happens if we switch from negative consumptions to positive ones, or vice versa.
        chunk_charge_loads = calculate_next_chunk(prev_consumption_sign, agent_id, timestep, remaining_battery_capacity,
                                                  soc, live_learner)
        predicted_chunk_size = len(chunk_charge_loads)
        step_in_chunk = 0
        consumption_sign = -prev_consumption_sign

    else:  # We already calculated the actions for this positive/negative consumption chunk.
        predicted_chunk_size = prev_predicted_chunk_size
        step_in_chunk += 1
        consumption_sign = prev_consumption_sign

    charge_load = -1 * consumption_sign * chunk_charge_loads[step_in_chunk]
    action = charge_load / remaining_battery_capacity

    return action, chunk_charge_loads, step_in_chunk, consumption_sign, predicted_chunk_size


class PredConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.remaining_battery_capacity = {}
        self.soc = {}
        self.consumption_sign = {}
        self.chunk_charge_loads = {}
        self.steps_in_chunk = {}
        self.predicted_chunk_size = {}

        self.live_learners = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.remaining_battery_capacity[agent_id] = 6.4
        self.soc[agent_id] = 0
        self.consumption_sign[agent_id] = -1
        self.chunk_charge_loads[agent_id] = [0]
        self.steps_in_chunk[agent_id] = 0
        self.predicted_chunk_size[agent_id] = 0

        if str(agent_id) not in self.live_learners:
            self.live_learners[str(agent_id)] = LiveLearner(300, 1)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        building_timestep = self.timestep // len(observation)

        action_out, self.chunk_charge_loads[agent_id], self.steps_in_chunk[agent_id], self.consumption_sign[agent_id], self.predicted_chunk_size[agent_id] = \
            pred_consumption_policy(observation[agent_id], building_timestep, agent_id,
                                    self.remaining_battery_capacity[agent_id],
                                    self.soc[agent_id], self.consumption_sign[agent_id],
                                    self.chunk_charge_loads[agent_id],
                                    self.steps_in_chunk[agent_id], self.live_learners[str(agent_id)], self.predicted_chunk_size[agent_id])

        energy = n.energy_normed(action_out * self.remaining_battery_capacity[agent_id], 5)
        efficiency = n.efficiency(energy, 5)

        previous_soc = self.soc[agent_id]
        self.soc[agent_id] = n.soc(energy, previous_soc, efficiency, self.remaining_battery_capacity[agent_id])

        battery_cons = n.last_energy_balance(self.soc[agent_id], previous_soc, efficiency)
        self.remaining_battery_capacity[agent_id] = n.new_capacity(self.remaining_battery_capacity[agent_id],
                                                                   battery_cons)

        return np.array([action_out], dtype=self.action_space[agent_id].dtype)
