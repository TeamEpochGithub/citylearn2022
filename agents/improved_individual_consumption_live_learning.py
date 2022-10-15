import csv
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

from agents.helper_classes.live_learning import LiveLearner
from data import citylearn_challenge_2022_phase_1 as competition_data

from traineval.training.data_preprocessing.find_action_limit import find_efficiency
from traineval.training.data_preprocessing import net_electricity_consumption as n
from traineval.training.data_preprocessing.pricing_simplified import pricing, shift_date

consumptions_path = osp.join(osp.dirname(competition_data.__file__), "consumptions/s_consumptions.csv")
# carbon_path = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")

consumptions = pd.read_csv(consumptions_path)[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]


# carbon = pd.read_csv(carbon_path)["kg_CO2/kWh"]
# carbon = carbon.values.tolist()[1:]


def individual_consumption_policy(observation, time_step, agent_id, capacity, soc, pos_in, energies_in, steps_in,
                                  live_learner):

    if time_step >= 8759:
        return 0, energies_in, steps_in, pos_in

    live_learner.update_lists(observation)

    consumption = consumptions[agent_id][time_step]

    # if time_step > 60:
    #     consumption = live_learner.get_consumption(1)
    #     consumption = consumption.item()

    hour = observation[2]
    date = shift_date(hour, observation[1], observation[0], shifts=1)

    if consumption * pos_in < 0:
        chunk = []
        steps = 0
        pos = -1 * pos_in

        t = 0

        while consumptions[agent_id][time_step + t] * pos >= 0:
            consumption = consumptions[agent_id][time_step + t]
            chunk.append(consumption)
            t += 1
            if time_step + t == 8759:
                break

        # if time_step <= 60:
        #     while consumptions[agent_id][time_step + t] * pos >= 0:
        #         consumption = consumptions[agent_id][time_step + t]
        #         chunk.append(consumption)
        #         t += 1
        #         if time_step + t == 8759:
        #             break
        # else:
        #     while live_learner.get_consumption(time_step + t) * pos >= 0:
        #         consumption = live_learner.get_consumption(time_step + t)
        #         consumption = consumption.item()
        #         chunk.append(consumption)
        #         t += 1
        #         if time_step + t == 8759:
        #             break

        consumption_sum = sum(chunk)

        if pos == -1:
            if -1 * consumption_sum >= (capacity - soc) / np.sqrt(0.83):
                relative_consumption = [i / consumption_sum for i in chunk]
                energies = [i * (capacity - soc) / np.sqrt(0.83) for i in relative_consumption]

            else:
                energies = [-1 * i for i in chunk]

        else:
            prices = []
            # emissions = []

            for h in range(len(chunk)):
                prices.append(pricing(date[2], date[0], date[1]))
                date = shift_date(date[0], date[1], date[2], shifts=1)

                # emissions.append(carbon[time_step + h])

            consumption_price = [prices[i] * c for i, c in enumerate(chunk)]

            if consumption_sum >= soc * np.sqrt(0.83):
                local_soc = soc * np.sqrt(0.83)
                local_consumption_price = consumption_price[:]
                energies = [0] * len(chunk)

                while local_soc != 0:
                    removing = local_consumption_price[:]
                    peak = max(local_consumption_price)
                    peak_indexes = [i for i, cp in enumerate(local_consumption_price) if cp == peak]

                    for _ in peak_indexes:
                        removing.remove(peak)

                    if len(removing) == 0:
                        removing = [0]

                    difference = peak - max(removing)
                    consumption_difference = [difference / prices[i] for i in peak_indexes]

                    if local_soc >= sum(consumption_difference):
                        for i, cd in enumerate(consumption_difference):
                            energies[peak_indexes[i]] += cd
                            local_soc -= cd
                            local_consumption_price[peak_indexes[i]] -= difference

                    else:
                        relative_difference = [cd / sum(consumption_difference) for cd in consumption_difference]

                        for i, rd in enumerate(relative_difference):
                            energies[peak_indexes[i]] += local_soc * rd
                            local_consumption_price[peak_indexes[i]] -= rd * local_soc * prices[peak_indexes[i]]

                        local_soc = 0

            else:
                energies = chunk
        energy = -1 * pos * energies[0]

    else:
        pos = pos_in
        energies = energies_in
        steps = steps_in + 1
        energy = -1 * energies[steps] * pos

    action = energy / capacity


    if time_step >= 60:
        predicted_consumptions = live_learner.predict_multiple_consumption(10)
    else:
        predicted_consumptions = np.zeros(10)
    predicted_consumptions = list(predicted_consumptions)

    observation.append(action)
    row = observation + predicted_consumptions
    action_file_path = osp.join(osp.dirname(competition_data.__file__), 'perfect_actions.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()

    return action, energies, steps, pos


class ImprovedIndividualConsumptionLiveLearningAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.capacity = {}
        self.soc = {}
        self.pos = {}
        self.energies = {}
        self.steps = {}

        self.live_learners = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.capacity[agent_id] = 6.4
        self.soc[agent_id] = 0
        self.pos[agent_id] = -1
        self.energies[agent_id] = [0]
        self.steps[agent_id] = 0

        if str(agent_id) not in self.live_learners:
            self.live_learners[str(agent_id)] = LiveLearner(300)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        collaborative_timestep = self.timestep // 5
        print(collaborative_timestep)

        action_out, self.energies[agent_id], self.steps[agent_id], self.pos[agent_id] = individual_consumption_policy(
            observation, collaborative_timestep, agent_id, self.capacity[agent_id], self.soc[agent_id],
            self.pos[agent_id], self.energies[agent_id], self.steps[agent_id], self.live_learners[str(agent_id)])

        action = max(min(action_out, 5 / self.capacity[agent_id]), -5 / self.capacity[agent_id])
        energy = action * self.capacity[agent_id]
        efficiency = find_efficiency(action, 5, self.capacity[agent_id])

        previous_soc = self.soc[agent_id]
        self.soc[agent_id] = n.soc(energy, previous_soc, efficiency, self.capacity[agent_id])

        battery_cons = n.last_energy_balance(self.soc[agent_id], previous_soc, efficiency)
        self.capacity[agent_id] = n.new_capacity(self.capacity[agent_id], battery_cons)

        return np.array([action_out], dtype=self.action_space[agent_id].dtype)
