import csv
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


def write_step_to_file(agent_id, timestep, action, observation):
    # ID, Action, Battery level, Consumption, Load, Solar, Carbon, Price
    row = [agent_id, timestep, action, observation[22], observation[23], observation[20], observation[21],
           observation[19],
           observation[24]]
    action_file_path = osp.join(osp.dirname(analysis_data.__file__), 'known_performance.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()


def write_historic_consumptions_to_file(agent_id, timestep):
    num_consumptions = 10
    row = []
    for i in range(num_consumptions):
        if timestep - i - 2 >= 0:
            row.append(consumptions[agent_id][timestep - i - 2])
        else:
            row.append(0)

    action_file_path = osp.join(osp.dirname(analysis_data.__file__), 'historic_consumptions.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()


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


def extra_charge(remaining_battery_capacity, soc, chunk_consumptions, chunk_charge_loads_in, date):
    chunk_total_consumption = sum(chunk_consumptions)
    chunk_charge_loads = chunk_charge_loads_in

    remaining_possible_charge = (remaining_battery_capacity - soc) / np.sqrt(0.83) + chunk_total_consumption

    consumption_prices, prices = get_consumption_prices(date, chunk_consumptions)

    price_occurrences = list(set(prices))
    price_indexes = [[i for i, p in enumerate(prices) if p == p_occurrence] for p_occurrence in
                     price_occurrences]

    if len(price_indexes) == 2:

        for i, price_occurrence_indexes in enumerate(price_indexes):

            if i == 0:
                opposite_index = 1
            elif i == 1:
                opposite_index = 0

            for price_index in price_occurrence_indexes:
                chunk_charge_loads[price_index] += remaining_possible_charge / \
                                                   (len(price_occurrence_indexes) +
                                                    len(price_indexes[opposite_index]) *
                                                    (price_occurrences[i] / price_occurrences[opposite_index]))

    else:
        chunk_charge_loads = [c + remaining_possible_charge / len(chunk_charge_loads_in) for c in
                              chunk_charge_loads_in]

    return chunk_charge_loads


def negative_consumption_scenario(date, chunk_consumptions, remaining_battery_capacity, soc):
    chunk_total_consumption = sum(chunk_consumptions)

    if -1 * chunk_total_consumption >= (remaining_battery_capacity - soc) / np.sqrt(0.83):
        # If more energy can be obtained than the one necessary to charge the battery
        relative_consumption = [i / chunk_total_consumption for i in chunk_consumptions]
        chunk_charge_loads = [i * (remaining_battery_capacity - soc) / np.sqrt(0.83) for i in relative_consumption]

    else:  # Otherwise charge with all the possible energy
        chunk_charge_loads = [-1 * i for i in chunk_consumptions]

        if -chunk_total_consumption >= 0.25 * ((remaining_battery_capacity - soc) / np.sqrt(0.83)):
            chunk_charge_loads = extra_charge(remaining_battery_capacity, soc, chunk_consumptions, chunk_charge_loads,
                                              date)

    return chunk_charge_loads


def lowering_peaks(local_soc, chunk_charge_loads, consumption_prices, prices):
    while local_soc != 0:

        # Get the peak consumption_price and check in which step the peak(s) happens
        max_consumption_price = max(consumption_prices)
        peak_indices = [i for i, p in enumerate(consumption_prices) if p == max_consumption_price]

        # List of other prices which do not indicate a peak
        consumption_prices_without_peak = [x for x in consumption_prices if x != max_consumption_price]

        if len(consumption_prices_without_peak) == 0:
            consumption_prices_without_peak = [0]

        # Get the difference in consumption price between the highest peak and the next highest peak
        # Make a list of the differences in consumption between the highest peaks and the next highest peak
        difference_from_peak = max_consumption_price - max(consumption_prices_without_peak)
        consumption_difference = [difference_from_peak / prices[i] for i in peak_indices]

        # Lower peaks to next highest peak
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

    for i in range(2, len(chunk_charge_loads)):
        if chunk_charge_loads[i] != 0 and chunk_charge_loads[i - 1] == 0 and chunk_charge_loads[i - 2] == 0:
            chunk_charge_loads[i - 2] = 0.000000001
            chunk_charge_loads[i - 1] = -0.0000001
            break

    return chunk_charge_loads


def get_consumption_prices(obs_date, chunk_consumptions):
    date = obs_date
    prices = []

    for hour in range(len(chunk_consumptions)):
        prices.append(pricing(date[2], date[0], date[1]))
        date = shift_date(date[0], date[1], date[2], shifts=1)

    consumption_prices = [prices[i] * c for i, c in enumerate(chunk_consumptions)]

    return consumption_prices, prices


def positive_consumption_scenario(date, chunk_consumptions, soc):
    chunk_total_consumption = sum(chunk_consumptions)

    if chunk_total_consumption >= soc * np.sqrt(0.83):
        # If fully discharging the battery doesn't bring the consumption to 0, we take the highest
        # price*consumption value and bring it down to the next highest price*consumption by reducing the
        # consumption at that time step. We do this consecutively until the battery has been emptied.

        consumption_prices, prices = get_consumption_prices(date, chunk_consumptions)

        local_soc = soc * np.sqrt(0.83)
        chunk_charge_loads = [0] * len(chunk_consumptions)

        return lowering_peaks(local_soc, chunk_charge_loads, consumption_prices, prices)

    else:
        return chunk_consumptions


def calculate_next_chunk(consumption_sign, agent_id, timestep, remaining_battery_capacity, soc, date):

    chunk_consumptions = get_chunk_consumptions(agent_id, timestep, consumption_sign)

    if consumption_sign == -1:  # If negative consumption
        chunk_charge_loads = negative_consumption_scenario(date, chunk_consumptions, remaining_battery_capacity, soc)
    else:
        chunk_charge_loads = positive_consumption_scenario(date, chunk_consumptions, soc)

    return chunk_charge_loads


def individual_consumption_policy(observation, timestep, agent_id, remaining_battery_capacity, soc, write_to_file):

    if timestep >= 8758:
        return -1

    next_consumption = consumptions[agent_id][timestep]

    if next_consumption == 0:
        return 0
    elif next_consumption > 0:
        consumption_sign = 1
    else:
        consumption_sign = -1

    date = shift_date(observation[2], observation[1], observation[0], shifts=1)

    chunk_charge_loads = calculate_next_chunk(consumption_sign, agent_id, timestep,
                                              remaining_battery_capacity, soc, date)

    charge_load = -1 * consumption_sign * chunk_charge_loads[0]
    action = charge_load / remaining_battery_capacity

    if write_to_file:
        write_step_to_file(agent_id, timestep, action, observation)
        write_historic_consumptions_to_file(agent_id, timestep)

    return action


class TimeStepKnownConsumptionAgentPeak:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.remaining_battery_capacity = {}
        self.soc = {}
        self.write_to_file = False

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.remaining_battery_capacity[agent_id] = 6.4
        self.soc[agent_id] = 0

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        building_timestep = self.timestep // len(observation)
        observation = observation[agent_id]

        action_out = \
            individual_consumption_policy(observation, building_timestep, agent_id,
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
