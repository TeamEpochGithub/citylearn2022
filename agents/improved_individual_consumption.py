import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from traineval.training.data_preprocessing.find_action_limit import find_efficiency
from traineval.training.data_preprocessing import net_electricity_consumption as n
from traineval.training.data_preprocessing.pricing_simplified import pricing, shift_date


consumptions = pd.read_csv("C:/Users/bjorn/OneDrive/Documents/TU Delft/EPOCH/citylearn-2022-starter-kit/data/citylearn_challenge_2022_phase_1/consumptions/s_consumptions.csv")[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]

carbon = pd.read_csv("C:/Users/bjorn/OneDrive/Documents/TU Delft/EPOCH/citylearn-2022-starter-kit/data/citylearn_challenge_2022_phase_1/carbon_intensity.csv")["kg_CO2/kWh"]
carbon = carbon.values.tolist()[1:]


def individual_consumption_policy(observation, time_step, agent_id, capacity, soc, pos_in, energies_in, steps_in):

    if time_step >= 8759:
        return 0, energies_in, steps_in, pos_in

    consumption_print = consumptions[agent_id][time_step]
    consumption = consumptions[agent_id][time_step]

    hour = observation[2]
    date = shift_date(hour, observation[1], observation[0], shifts=1)

    if consumption * pos_in < 0:
        chunk = []
        steps = 0
        pos = -1*pos_in

        t = 0

        while consumption * pos >= 0 and time_step + t + 1 <= 8758:
            chunk.append(consumption)
            t += 1
            print(f"Whyyyy {time_step + t}")
            consumption = consumptions[agent_id][time_step + t]

        if time_step + t + 1 == 8758:
            chunk.append(consumption)

        if pos == -1:
            consumption_sum = sum(chunk)
            relative_consumption = [i / consumption_sum for i in chunk]
            energies = [i * (capacity - soc)/np.sqrt(0.83) for i in relative_consumption]
            energy = energies[0]

        else:
            prices = []
            emissions = []

            for h in range(len(chunk)):
                prices.append(pricing(date[2], date[0], date[1]))
                date = shift_date(date[0], date[1], date[2], shifts=1)

                emissions.append(carbon[time_step + h])

            consumption_price = [prices[i] * c for i, c in enumerate(chunk)]
            consumption_emission = [emissions[i] * c for i, c in enumerate(chunk)]




            local_soc = soc * np.sqrt(0.83)
            local_consumption_price = consumption_price
            energies = [0] * len(consumption_price)
            if agent_id == 2:
                print(f"List: {consumption_price}")

            if -1*sum(chunk) >= soc * np.sqrt(0.83):

                while not local_soc == 0:

                    removing = local_consumption_price
                    peak = max(local_consumption_price)
                    peak_indexes = []

                    for i, cp in enumerate(local_consumption_price):
                        if cp == peak:
                            peak_indexes.append(i)
                            removing.pop(i)
                    print(f"Removing: {removing} Agent: {agent_id}")
                    difference = peak - max(removing)
                    consumption_difference = [difference/prices[i] for i in peak_indexes]

                    if local_soc - sum(consumption_difference) >= 0:
                        for i, cd in enumerate(consumption_difference):
                            energies[peak_indexes[i]] += cd
                            local_soc -= cd
                            local_consumption_price[peak_indexes[i]] -= difference

                    else:
                        relative_difference = [cd/sum(consumption_difference) for cd in consumption_difference]

                        for i, cd in enumerate(relative_difference):
                            energies[peak_indexes[i]] += local_soc*cd
                            local_consumption_price[peak_indexes[i]] -= cd * local_soc * prices[peak_indexes[i]]

                        local_soc = 0

            else:

                price_sum = sum(consumption_price)
                relative_price = [i / price_sum for i in consumption_price]

                energies = [i * soc * np.sqrt(0.83) for i in relative_price]

            energy = -1*energies[0]

    else:
        pos = pos_in
        energies = energies_in
        steps = steps_in + 1

        energy = -1*energies[steps]*pos

    action = energy/capacity

    if agent_id == 0:
        print([f"Agent {agent_id}, Action: {action}, Energy: {energy}, Consumption: {consumption_print}, Time: {time_step}, SOC observed: {observation[22]}"])

    return action, energies, steps, pos


class IndividualConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.capacity = {}
        self.soc = {}
        self.pos = {}
        self.energies = {}
        self.steps = {}

        self.plot = {}


    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.capacity[agent_id] = 6.4
        self.soc[agent_id] = 0
        self.pos[agent_id] = -1
        self.energies[agent_id] = [0]
        self.steps[agent_id] = 0

        self.plot[agent_id] = [[], [], []]

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        collaborative_timestep = self.timestep//5


        if agent_id == 0:
            print(f"and an actual net consumption {observation[23]}")


        if collaborative_timestep > 0:
            self.plot[agent_id][0].append(observation[23])
            self.plot[agent_id][1].append(observation[20]-observation[21])
            self.plot[agent_id][2].append((observation[20]-observation[21])*observation[24])

        if collaborative_timestep == 48 and agent_id == 0:
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][0], color="red")
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][1], color="blue")
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][2], color="green")
            plt.show()
            print(self.plot[agent_id])


        action_out, self.energies[agent_id], self.steps[agent_id], self.pos[agent_id] = individual_consumption_policy(observation, collaborative_timestep, agent_id, self.capacity[agent_id], self.soc[agent_id], self.pos[agent_id], self.energies[agent_id], self.steps[agent_id])

        action = max(min(action_out, 5 / self.capacity[agent_id]), -5 / self.capacity[agent_id])
        energy = action*self.capacity[agent_id]
        efficiency = find_efficiency(action, 5, self.capacity[agent_id])

        previous_soc = self.soc[agent_id]
        self.soc[agent_id] = n.soc(energy, previous_soc, efficiency, self.capacity[agent_id])


        if agent_id == 0:
            print(f"This gives a new SOC of: {self.soc[agent_id]}")


        battery_cons = n.last_energy_balance(self.soc[agent_id], previous_soc, efficiency)
        self.capacity[agent_id] = n.new_capacity(self.capacity[agent_id], battery_cons)

        return np.array([action_out], dtype=self.action_space[agent_id].dtype)



