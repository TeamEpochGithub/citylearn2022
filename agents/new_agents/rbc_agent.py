import sys
import pandas as pd
import numpy as np

consumptions = pd.read_csv("../../data/citylearn_challenge_2022_phase_1/consumptions/s_consumptions.csv")[[f"{i}" for i in range(5)]]
consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]



def rbc_optimized_policy(observation, action_space):
    """
    The actions are designed such that the agent discharges the controlled storage system(s) by 2.0% of its maximum
    capacity every hour between 07:00 AM and 03:00 PM, discharges by 4.4% of its maximum capacity between 04:00 PM and
    06:00 PM, discharges by 2.4% of its maximum capacity between 07:00 PM and 10:00 PM, charges by 3.4% of its maximum
    capacity between 11:00 PM to midnight and charges by 5.532% of its maximum capacity at every other hour.
    """
    hour = observation[2]  # Hour index is 2 for all observations

    # Daytime: release stored energy  2*0.08 + 0.1*7 + 0.09
    if 7 <= hour <= 15:
        action = -0.02

    elif 16 <= hour <= 18:
        action = -0.0044

    elif 19 <= hour <= 22:
        action = -0.024

    # Early nightime: store DHW and/or cooling energy
    elif 23 <= hour <= 24:
        action = 0.034

    elif 1 <= hour <= 6:
        action = 0.05532

    else:
        action = 0.0

    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action


def rbc_policy(observation, action_space):
    """
    Simple rule based policy based on day or night time
    """
    hour = observation[2]  # Hour index is 2 for all observations

    action = 0.0  # Default value
    if 9 <= hour <= 21:
        # Daytime: release stored energy
        action = -0.08
    elif (1 <= hour <= 8) or (22 <= hour <= 24):
        # Early nightime: store DHW and/or cooling energy
        action = 0.091

    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action


def battery_rbc_policy(observation, action_space):
    """

        The actions are optimized for electrical storage (battery) such that the agent charges the controlled storage
        system(s) by 11.0% of its maximum capacity every hour between 06:00 AM and 02:00 PM, and discharges 6.7% of its
        maximum capacity at every other hour.
    """
    hour = observation[2]  # Hour index is 2 for all observations

    action = -0.067  # Early morning and late evening: release energy
    if 6 <= hour <= 14:
        # Late morning and early evening: store energy
        action = 0.11

    action = np.array([action], dtype=action_space.dtype)

    return action

def nothing_policy(observation, action_space, agent_id):

    if observation[2]%2 == 0:
        action = 1
    else:
        action = -1

    action = np.array([action], dtype=action_space.dtype)

    return action

def smart_policy(action_space, time_step, agent_id):
    consumption = consumptions[agent_id][time_step]
    # hour = observation[2]
    action = -consumption/6.4
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



class BasicRBCAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return nothing_policy(observation, self.action_space[agent_id], agent_id)




class BasicRBCAgent2:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}
        self.timestep = -1

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        self.timestep += 1
        collaborative_timestep = self.timestep//5
        return smart_policy(self.action_space[agent_id], collaborative_timestep, agent_id)



