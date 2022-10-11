import sys
import pandas as pd
import numpy as np

consumptions = pd.read_csv("C:/Users/bjorn/OneDrive/Documents/TU Delft/EPOCH/citylearn-2022-starter-kit/data/citylearn_challenge_2022_phase_1/consumptions/s_consumptions.csv")[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]

def individual_consumption_policy(action_space, time_step, agent_id):
    consumption = consumptions[agent_id][time_step]

    action = -consumption/6.4

    # hour = observation[2]
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

    print([f"{agent_id} Action: {action}, Consumption: {consumption}, Time: {time_step}"])

    action = np.array([action], dtype=action_space.dtype)
    return action

class IndividualConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        self.timestep += 1
        collaborative_timestep = self.timestep//5
        return individual_consumption_policy(self.action_space[agent_id], collaborative_timestep, agent_id)



