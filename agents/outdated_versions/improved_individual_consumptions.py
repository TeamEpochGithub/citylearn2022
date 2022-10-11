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
carbon = carbon.values.tolist()[1:] + [0, 0]

def individual_consumption_policy(observation, action_space, time_step, agent_id, capacity, soc, soc_init, capacity_init, daily_prices, daily_emissions):
    consumption = consumptions[agent_id][time_step]
    hour = observation[2]
    date = shift_date(hour, observation[1], observation[0], shifts=1)

    daily_consumption = consumptions[agent_id][(time_step//24)*24:(time_step//24)*24 + 24]

    negative_consumption, negative_hour, positive_consumption, positive_hour = [], [], [], []

    if time_step > 8758:
        return np.array([0], dtype=action_space.dtype), daily_prices, daily_emissions

    for h, i in enumerate(daily_consumption):
        if i < 0:
            negative_consumption.append(-1*i)
            negative_hour.append(h)
        else:
            positive_consumption.append(i)
            positive_hour.append(h)

    if date[0]-1 in negative_hour:
        negative_sum = sum(negative_consumption)
        relative_neg_consumption = [i / negative_sum for i in negative_consumption]

        relative_neg_available = [i * (capacity_init - soc_init)/np.sqrt(0.83) for i in relative_neg_consumption]

        energy = relative_neg_available[negative_hour.index(date[0]-1)]
        action = energy/capacity

        consumption_price = daily_prices
        consumption_emission = daily_emissions

    else:
        if date[0] == 1:
            prices = []
            emissions = []

            for h in positive_hour:
                prices.append(pricing(date[2], h + 1, date[1]))
                emissions.append(carbon[time_step + h])

            consumption_price = [prices[i] * c for i, c in enumerate(positive_consumption)]
            consumption_emission = [emissions[i] * c for i, c in enumerate(positive_consumption)]

        else:
            consumption_price = daily_prices
            consumption_emission = daily_emissions

        price_sum = sum(consumption_price)
        relative_price = [i / price_sum for i in consumption_price]

        relative_price_available = [i * capacity_init for i in relative_price]

        energy = relative_price_available[positive_hour.index(date[0]-1)]
        action = -energy/capacity


    #
    # if consumption < 0:
    #     action = -consumption/6.4
    # else:
    #     action = 0
    # #     action = -1
    # # elif 20 < hour <= 24:
    # #     action = -0.5
    # # else:
    # #     action = 0



    if agent_id == 0:
        print([f"Agent {agent_id}, Action: {action}, Energy: {energy}, Consumption: {consumption}, Time: {time_step}, SOC observed: {observation[22]}"])


    action = np.array([action], dtype=action_space.dtype)
    return action, consumption_price, consumption_emission


class IndividualConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.capacity = {}
        self.soc = {}
        self.soc_init = {}
        self.capacity_init = {}
        self.daily_prices = {}
        self.daily_emissions = {}
        self.plot = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.capacity[agent_id] = 6.4
        self.soc[agent_id] = 0
        self.soc_init[agent_id] = 0
        self.capacity_init[agent_id] = 6.4
        self.daily_prices[agent_id] = []
        self.daily_emissions[agent_id] = []
        self.plot[agent_id] = [[], [], []]

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        collaborative_timestep = self.timestep//5


        if collaborative_timestep > 7970-24:
            self.plot[agent_id][0].append(observation[23])
            self.plot[agent_id][1].append(observation[20]-observation[21])
            self.plot[agent_id][2].append((observation[20]-observation[21])*observation[24])
        if collaborative_timestep == 7970+24 and agent_id == 0:
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][0], color="red")
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][1], color="blue")
            plt.plot(range(len(self.plot[agent_id][0])), self.plot[agent_id][2], color="green")
            plt.show()
            print(self.plot[agent_id])


        if observation[2] == 24:
            self.soc_init[agent_id] = self.soc[agent_id]
            self.capacity_init[agent_id] = self.capacity[agent_id]

        action_out, self.daily_prices[agent_id], self.daily_emissions[agent_id] = individual_consumption_policy(observation, self.action_space[agent_id], collaborative_timestep, agent_id, self.capacity[agent_id], self.soc[agent_id], self.soc_init[agent_id], self.capacity_init[agent_id], self.daily_prices[agent_id], self.daily_emissions[agent_id])

        action = max(min(action_out[0], 5 / self.capacity[agent_id]), -5 / self.capacity[agent_id])
        energy = action*self.capacity[agent_id]
        efficiency = find_efficiency(action, 5, self.capacity[agent_id])

        previous_soc = self.soc[agent_id]
        self.soc[agent_id] = n.soc(energy, previous_soc, efficiency, self.capacity[agent_id])


        if agent_id == 0:
            print(f"This gives a new SOC of: {self.soc[agent_id]}")


        battery_cons = n.last_energy_balance(self.soc[agent_id], previous_soc, efficiency)
        self.capacity[agent_id] = n.new_capacity(self.capacity[agent_id], battery_cons)

        return action_out



