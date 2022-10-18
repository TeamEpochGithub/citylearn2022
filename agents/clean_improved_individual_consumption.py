import sys
import pandas as pd
import numpy as np
import os.path as osp
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

    while consumptions[agent_id][timestep + future_steps] * consumption_sign >= 0:  # consumptions with the same sign
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
        chunk_charge_loads = [i * (remaining_battery_capacity - soc) / np.sqrt(0.83) for i in relative_consumption]
    else:  # Otherwise charge with all the possible energy
        chunk_charge_loads = [-1 * i for i in chunk_consumptions]

    return chunk_charge_loads


def calculate_prices(observation, step_range):
    prices = []
    date = shift_date(observation[2], observation[1], observation[0], shifts=1)

    for h in range(step_range):
        prices.append(pricing(date[2], date[0], date[1]))
        date = shift_date(date[0], date[1], date[2], shifts=1)

    return prices


def positive_consumption_scenario(prices, chunk_consumptions, soc, consumption_prices):
    chunk_total_consumption = sum(chunk_consumptions)

    # consumption_prices = calculate_consumption_prices()

    if chunk_total_consumption >= soc * np.sqrt(0.83):
        # If fully discharging the battery doesn't bring the consumption to 0, we take the highest
        # price*consumption value and bring it down to the next highest price*consumption by reducing the
        # consumption at that time step. We do this consecutively until the battery has been emptied.

        local_soc = soc * np.sqrt(0.83)
        chunk_charge_loads = [0] * len(chunk_consumptions)

        while local_soc != 0:
            max_consumption_price = max(consumption_prices)
            peak_indices = [i for i, p in enumerate(consumption_prices) if p == max_consumption_price]

            consumption_prices_without_peak = [x for x in consumption_prices if x != max_consumption_price]

            if len(consumption_prices_without_peak) == 0:
                consumption_prices_without_peak = [0]

            difference_from_peak = max_consumption_price - max(consumption_prices_without_peak)
            consumption_difference = [difference_from_peak / prices[i] for i in peak_indices]

            if local_soc >= sum(consumption_difference):
                for i, difference in enumerate(consumption_difference):
                    chunk_charge_loads[peak_indices[i]] += difference
                    local_soc -= difference
                    consumption_prices[peak_indices[i]] -= difference_from_peak
            else:
                relative_difference = [c / sum(consumption_difference) for c in consumption_difference]

                for i, rd in enumerate(relative_difference):
                    chunk_charge_loads[peak_indices[i]] += local_soc * rd
                    consumption_prices[peak_indices[i]] -= rd * local_soc * prices[peak_indices[i]]

                local_soc = 0

    else:
        chunk_charge_loads = chunk_consumptions

    return chunk_charge_loads


def calculate_next_chunk(observation, prev_consumption_sign, agent_id, timestep, remaining_battery_capacity, soc):
    consumption_sign = -prev_consumption_sign

    chunk_consumptions = get_chunk_consumptions(agent_id, timestep, consumption_sign)

    if consumption_sign == -1:  # If negative consumption
        chunk_charge_loads = negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc)
    else:
        prices = calculate_prices(observation, len(chunk_consumptions))
        consumption_prices = [prices[i] * c for i, c in enumerate(chunk_consumptions)]
        chunk_charge_loads = positive_consumption_scenario(prices, chunk_consumptions, soc, consumption_prices)

    return chunk_charge_loads


def individual_consumption_policy(observation,
                                  timestep,
                                  agent_id,
                                  remaining_battery_capacity,
                                  soc,
                                  prev_consumption_sign,
                                  chunk_charge_loads,
                                  step_in_chunk):

    if timestep >= 8759:
        return 0, chunk_charge_loads, step_in_chunk, prev_consumption_sign

    next_consumption = consumptions[agent_id][timestep]

    if next_consumption * prev_consumption_sign < 0:
        # If next consumption is positive, and previous was negative.
        # Or if, next consumption is negative, and previous was negative.
        # This happens if we switch from negative consumptions to positive ones, or vice versa.
        chunk_charge_loads = calculate_next_chunk(observation=observation,
                                                  prev_consumption_sign=prev_consumption_sign,
                                                  agent_id=agent_id,
                                                  timestep=timestep,
                                                  remaining_battery_capacity=remaining_battery_capacity,
                                                  soc=soc)

        step_in_chunk = 0
        consumption_sign = -prev_consumption_sign

    else:  # We already calculated the actions for this positive/negative consumption chunk.
        step_in_chunk += 1
        consumption_sign = prev_consumption_sign

    charge_load = -1 * consumption_sign * chunk_charge_loads[step_in_chunk]
    action = charge_load / remaining_battery_capacity

    return action, chunk_charge_loads, step_in_chunk, consumption_sign


class CleanImprovedIndividualConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.remaining_battery_capacity = {}
        self.soc = {}
        self.pos = {}
        self.energies = {}
        self.steps = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.remaining_battery_capacity[agent_id] = 6.4
        self.soc[agent_id] = 0
        self.pos[agent_id] = -1
        self.energies[agent_id] = [0]
        self.steps[agent_id] = 0

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        collaborative_timestep = self.timestep // 5

        action_out, self.energies[agent_id], self.steps[agent_id], self.pos[agent_id] = individual_consumption_policy(
            observation, collaborative_timestep, agent_id, self.remaining_battery_capacity[agent_id],
            self.soc[agent_id], self.pos[agent_id], self.energies[agent_id], self.steps[agent_id])

        energy = n.energy_normed(action_out * self.remaining_battery_capacity[agent_id], 5)
        efficiency = n.efficiency(energy, 5)

        previous_soc = self.soc[agent_id]
        self.soc[agent_id] = n.soc(energy, previous_soc, efficiency, self.remaining_battery_capacity[agent_id])

        battery_cons = n.last_energy_balance(self.soc[agent_id], previous_soc, efficiency)
        self.remaining_battery_capacity[agent_id] = n.new_capacity(self.remaining_battery_capacity[agent_id],
                                                                   battery_cons)

        return np.array([action_out], dtype=self.action_space[agent_id].dtype)
