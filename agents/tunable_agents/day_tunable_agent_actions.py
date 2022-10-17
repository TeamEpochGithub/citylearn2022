import numpy as np


def combined_policy(observation, action_space, args, day):
    month = observation[0]
    day_type = observation[1]
    hour = observation[2]

    if day != args["day"]:
        return np.array([0], dtype=action_space.dtype)

    action = args[f"hour_{int(hour)}"]

    action = np.array([action], dtype=action_space.dtype)

    return action


class TunableDayActionsAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self, args):
        self.action_space = {}
        self.args = args

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id, day):
        """Get observation return action"""
        return combined_policy(observation, self.action_space[agent_id], self.args, day)
