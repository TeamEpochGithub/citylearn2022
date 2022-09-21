import numpy as np
import sys


class BasicRBCAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def rbc_policy(self, observation, action_space):
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

    def compute_action(self, observation, building_count):
        """Get observation return action"""

        action_list = []

        for current_building in range(building_count):
            action_list.append(self.rbc_policy(observation, self.action_space[current_building]))

        return action_list
