import sys

import numpy as np

def rbc_policy(observation, action_space):
    """
    Simple rule based policy based on day or night time
    """
    hour = observation[2] # Hour index is 2 for all observations

    # print(observation)
    # sys.exit()

    if hour >= 7 and hour <= 15:
        action = -0.02

    elif hour >= 16 and hour <= 18:
        action = -0.0044

    elif hour >= 19 and hour <= 22:
        action = -0.024

        # Early nightime: store DHW and/or cooling energy
    elif hour >= 23 and hour <= 24:
        action = 0.034

    elif hour >= 1 and hour <= 6:
        action = 0.05532

    else:
        action = 0.0

    # action = 0.0 # Default value
    # if 9 <= hour <= 21:
    #     # Daytime: release stored energy
    #     action = -0.08
    # elif (1 <= hour <= 8) or (22 <= hour <= 24):
    #     # Early nightime: store DHW and/or cooling energy
    #     action = 0.091

    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action



    #
    # self.actions = actions
    # self.next_time_step()
    # return actions

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
        return rbc_policy(observation, self.action_space[agent_id])