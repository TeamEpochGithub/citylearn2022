import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
from data import citylearn_challenge_2022_phase_1 as competition_data

from traineval.training.data_preprocessing.find_action_limit import find_efficiency
from traineval.training.data_preprocessing import net_electricity_consumption as n
from traineval.training.data_preprocessing.pricing_simplified import pricing, shift_date


consumptions_path = osp.join(osp.dirname(competition_data.__file__), "consumptions/s_consumptions.csv")
carbon_path = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")

consumptions = pd.read_csv(consumptions_path)[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]

carbon = pd.read_csv(carbon_path)["kg_CO2/kWh"]
carbon = carbon.values.tolist()[1:]


def individual_consumption_policy(observation, time_step, agent_id, capacity, soc, pos_in, energies_in, steps_in):

    if time_step >= 8759:
        return 0, energies_in, steps_in, pos_in

    next_consumption = consumptions[agent_id][time_step]

    hour = observation[2]
    date = shift_date(hour, observation[1], observation[0], shifts=1)

    if next_consumption * pos_in < 0: # pos is 1 if previous consumption was positive and -1 if it was negative
        chunk_consumptions = []
        steps = 0
        pos = -pos_in

        t = 0

        while consumptions[agent_id][time_step + t] * pos >= 0: # consumptions with the same sign
            next_consumption = consumptions[agent_id][time_step + t]
            chunk_consumptions.append(next_consumption)
            t += 1

            if time_step + t >= 8759:
                break

        chunk_total_consumption = sum(chunk_consumptions)

        if pos == -1: # If negative consumption
            if -1 * chunk_total_consumption >= (capacity - soc) / np.sqrt(0.83): # If more energy can be obtained than the one necessary to charge the battery
                relative_consumption = [i / chunk_total_consumption for i in chunk_consumptions]
                energies = [i * (capacity - soc) / np.sqrt(0.83) for i in relative_consumption]
            else: # Otherwise charge with all the possible energy
                energies = [-1*i for i in chunk_consumptions]

        else:
            prices = []
            emissions = []

            for h in range(len(chunk_consumptions)):
                prices.append(pricing(date[2], date[0], date[1]))
                print(prices)

                date = shift_date(date[0], date[1], date[2], shifts=1)

                emissions.append(carbon[time_step + h])

            consumption_prices = [prices[i] * c for i, c in enumerate(chunk_consumptions)]

            if chunk_total_consumption >= soc * np.sqrt(0.83):
                # If fully discharging the battery doesn't bring the consumption to 0, we take the highest
                # price*consumption value and bring it down to the next highest price*consumption by reducing the
                # consumption at that time step. We do this consecutively until the battery has been emptied.

                local_soc = soc * np.sqrt(0.83)
                energies = [0] * len(chunk_consumptions)

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
                            energies[peak_indices[i]] += difference
                            local_soc -= difference
                            consumption_prices[peak_indices[i]] -= difference_from_peak
                    else:
                        relative_difference = [c / sum(consumption_difference) for c in consumption_difference]

                        for i, rd in enumerate(relative_difference):
                            energies[peak_indices[i]] += local_soc*rd
                            consumption_prices[peak_indices[i]] -= rd * local_soc * prices[peak_indices[i]]

                        local_soc = 0

            else:
                energies = chunk_consumptions
        energy = -1 * pos * energies[0]

    else: # We already calculated the actions for this positive/negative consumption chunk.
        pos = pos_in
        energies = energies_in
        steps = steps_in + 1

        energy = -1*energies[steps]*pos

    action = energy / capacity

    # observation.append(action)
    # row = observation
    # action_file_path = osp.join(osp.dirname(competition_data.__file__), 'perfect_actions.csv')
    # action_file = open(action_file_path, 'a', newline="")
    # writer = csv.writer(action_file)
    # writer.writerow(row)
    # action_file.close()

    return action, energies, steps, pos


class ImprovedIndividualConsumptionAgentOld:

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
        collaborative_timestep = self.timestep//5

        action_out, self.energies[agent_id], self.steps[agent_id], self.pos[agent_id] = individual_consumption_policy(observation, collaborative_timestep, agent_id, self.remaining_battery_capacity[agent_id], self.soc[agent_id], self.pos[agent_id], self.energies[agent_id], self.steps[agent_id])

        # max_power = n.max_power(self.soc[agent_id], 5, self.capacity[agent_id])
        energy = n.energy_normed(action_out * self.remaining_battery_capacity[agent_id], 5)
        efficiency = n.efficiency(energy, 5)

        previous_soc = self.soc[agent_id]
        self.soc[agent_id] = n.soc(energy, previous_soc, efficiency, self.remaining_battery_capacity[agent_id])

        battery_cons = n.last_energy_balance(self.soc[agent_id], previous_soc, efficiency)
        self.remaining_battery_capacity[agent_id] = n.new_capacity(self.remaining_battery_capacity[agent_id], battery_cons)

        # print(action_out)

        return np.array([action_out], dtype=self.action_space[agent_id].dtype)
