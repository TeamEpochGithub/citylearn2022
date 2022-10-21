import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from data import citylearn_challenge_2022_phase_1 as competition_data

from traineval.training.data_preprocessing import net_electricity_consumption as n
from traineval.training.data_preprocessing.pricing_simplified import pricing, shift_date

consumptions_path = osp.join(osp.dirname(competition_data.__file__), "consumptions/building_consumptions.csv")
carbon_path = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")

consumptions = pd.read_csv(consumptions_path)[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]


carbon = pd.read_csv(carbon_path)["kg_CO2/kWh"]
carbon = carbon.values.tolist()[1:]

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
        chunk_charge_loads = [i * (remaining_battery_capacity - soc) / np.sqrt(0.83) for i in relative_consumption]
    else:  # Otherwise charge with all the possible energy
        chunk_charge_loads = [-1 * i for i in chunk_consumptions]

    return chunk_charge_loads


def positive_consumption_scenario(observation, chunk_consumptions, timestep, soc, agent_id):
    chunk_total_consumption = sum(chunk_consumptions)

    if chunk_total_consumption >= soc * np.sqrt(0.83):
        # If fully discharging the battery doesn't bring the consumption to 0, we take the highest
        # price*consumption value and bring it down to the next highest price*consumption by reducing the
        # consumption at that time step. We do this consecutively until the battery has been emptied.

        date = shift_date(observation[2], observation[1], observation[0], shifts=1)

        prices = []
        emissions = []

        for hour in range(len(chunk_consumptions)):
            prices.append(pricing(date[2], date[0], date[1]))
            date = shift_date(date[0], date[1], date[2], shifts=1)

            emissions.append(carbon[timestep + hour])

        consumption_prices = [prices[i] * c for i, c in enumerate(chunk_consumptions)]
        consumption_emissions = [emissions[i] * c for i, c in enumerate(chunk_consumptions)]

        max_p, min_p, max_e, min_e = max(consumption_prices), min(consumption_prices), max(consumption_emissions), min(consumption_emissions)
        if len(chunk_consumptions) == 1:
            min_p, min_e = 0, 0
        scaled_consumption_prices = [(i - min_p)/(max_p-min_p) for i in consumption_prices]
        scaled_consumption_emissions = [(i - min_e)/(max_e-min_e) for i in consumption_emissions]

        weight_p = 1
        weight_e = 1-weight_p
        reference_curve = [(weight_p * scaled_consumption_prices[i] + weight_e * scaled_consumption_emissions[i]) for i in range(len(chunk_consumptions))]

        local_soc = soc * np.sqrt(0.83)
        chunk_charge_loads = [0] * len(chunk_consumptions)

        return lowering_peaks(local_soc, chunk_charge_loads, reference_curve, prices, emissions, max_p, min_p, max_e, min_e, weight_p, weight_e, timestep, agent_id)

    else:
        return chunk_consumptions


def lowering_peaks(local_soc, chunk_charge_loads, reference_curve, prices, emissions, max_p, min_p, max_e, min_e, weight_p, weight_e, timestep, agent_id):
    while local_soc != 0:

        # Get the peak consumption_price and check in which step the peak(s) happens
        peak = max(reference_curve)
        peak_indices = [i for i, p in enumerate(reference_curve) if p == peak]

        # List of other prices which do not indicate a peak
        reference_curve_without_peak = [x for x in reference_curve if x != peak]
        if len(reference_curve_without_peak) == 0:
            reference_curve_without_peak = [((-1*weight_p*min_p)/(max_p-min_p))-((weight_e*min_e)/(max_e-min_e))]

        # Get the difference in consumption price between the highest peak and the next highest peak
        # Make a list of the differences in consumption between the highest peaks and the next highest peak
        difference_from_peak = peak - max(reference_curve_without_peak)
        consumption_difference = [find_consumption_difference(difference_from_peak, prices[i], emissions[i], max_p, min_p, max_e, min_e, weight_p, weight_e) for i in peak_indices]
        # Lower peaks to next highest peak
        if local_soc >= sum(consumption_difference):
            for i, difference in enumerate(consumption_difference):
                chunk_charge_loads[peak_indices[i]] += difference
                local_soc -= difference
                reference_curve[peak_indices[i]] -= difference_from_peak
        else:
            relative_difference = [c / sum(consumption_difference) for c in consumption_difference]

            for i, rd in enumerate(relative_difference):
                chunk_charge_load = rd * local_soc
                chunk_charge_loads[peak_indices[i]] += chunk_charge_load

                reference_curve[peak_indices[i]] -= (chunk_charge_load * prices[peak_indices[i]] - min_p)/(max_p-min_p) + (chunk_charge_load*emissions[peak_indices[i]] - min_e)/(max_e-min_e)

            local_soc = 0

    for i in range(2, len(chunk_charge_loads)):
        if chunk_charge_loads[i] != 0 and chunk_charge_loads[i - 1] == 0 and chunk_charge_loads[i - 2] == 0:
            chunk_charge_loads[i - 2] = 0.000000001
            chunk_charge_loads[i - 1] = -0.0000001

    return chunk_charge_loads


def find_consumption_difference(difference_from_peak, price, emission, max_p, min_p, max_e, min_e, weight_p, weight_e):
    range_p = max_p - min_p
    range_e = max_e - min_e

    consumption = (difference_from_peak * range_e * range_p)\
        / ((weight_p * range_e * price) + (weight_e * range_p * emission))

    return consumption


def calculate_next_chunk(observation, consumption_sign, agent_id, timestep, remaining_battery_capacity, soc):

    chunk_consumptions = get_chunk_consumptions(agent_id, timestep, consumption_sign)
    if consumption_sign == -1:  # If negative consumption
        chunk_charge_loads = negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc)
    else:
        chunk_charge_loads = positive_consumption_scenario(observation, chunk_consumptions, timestep, soc, agent_id)

    return chunk_charge_loads


def individual_consumption_policy(observation, timestep, agent_id, remaining_battery_capacity, soc):
    if timestep >= 8759:
        return 0

    next_consumption = consumptions[agent_id][timestep]
    if next_consumption == 0:
        return 0

    elif next_consumption > 0:
        consumption_sign = 1
    else:
        consumption_sign = -1

    chunk_charge_loads = calculate_next_chunk(observation, consumption_sign, agent_id, timestep, remaining_battery_capacity,
                                              soc)

    charge_load = -1 * consumption_sign * chunk_charge_loads[0]
    action = charge_load / remaining_battery_capacity

    return action



class TimeStepKnownConsumptionAgentPeakCarbon:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.remaining_battery_capacity = {}
        self.soc = {}
        self.plot = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.remaining_battery_capacity[agent_id] = 6.4
        self.soc[agent_id] = 0
        self.plot[agent_id] = [[], [], [], []]

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        building_timestep = self.timestep // len(observation)
        observation = observation[agent_id]

        if building_timestep > 24:
            self.plot[agent_id][0].append(observation[23])
            self.plot[agent_id][1].append(observation[20] - observation[21])
            self.plot[agent_id][2].append((observation[20] - observation[21]) * observation[24])
            self.plot[agent_id][3].append(observation[23] * observation[24])

        if building_timestep == 72 and agent_id == 0:
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][0], color="red")
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][1], color="blue")
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][2], color="green")
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][3], color="yellow")
            plt.plot(range(len(self.plot[agent_id][0])), [0] * len(self.plot[agent_id][0]), color="black")
            plt.show()

        action_out = \
            individual_consumption_policy(observation, building_timestep, agent_id,
                                          self.remaining_battery_capacity[agent_id],
                                          self.soc[agent_id])

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
