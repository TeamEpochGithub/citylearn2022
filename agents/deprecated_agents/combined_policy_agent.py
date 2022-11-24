import sys

import numpy as np

from traineval.utils.convert_arguments import environment_convert_argument, environment_convert_scalars


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
    discharge_factor = -0.09
    charge_factor = 0.15
    hours_in = 7
    hours_out = 14

    action = discharge_factor  # Early morning and late evening: release energy
    if hours_in <= hour <= hours_out:
        # Late morning and early evening: store energy
        action = charge_factor

    action = np.array([action], dtype=action_space.dtype)

    return action


def grid_focused_policy(observation, action_space):
    consumption_index = environment_convert_argument(["net_electricity_consumption"])[0]
    consumption = observation[consumption_index]

    old_range = (5.2 - (0))
    new_range = (1 - (-1))
    pricing_scaled = (((consumption - (0)) * new_range) / old_range) + (-1)
    action = (-pricing_scaled) / 5

    action = np.array([action], dtype=action_space.dtype)

    return action


def price_focused_policy(observation, action_space):
    pricing_index = environment_convert_argument(["electricity_pricing"])[0]
    pricing = observation[pricing_index]

    old_range = (0.54 - (0.15))
    new_range = (1 - (-1))
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    pricing_scaled = (((pricing - (0.15)) * new_range) / old_range) + (-1)
    action = (-pricing_scaled) / 5

    action = np.array([action], dtype=action_space.dtype)

    return action


def carbon_focused_policy(observation, action_space):
    carbon_index = environment_convert_argument(["carbon_intensity"])[0]
    carbon = observation[carbon_index]

    old_range = (0.2818 - (0.0703))
    new_range = (1 - (-1))
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    carbon_scaled = (((carbon - (0.0703)) * new_range) / old_range) + (-1)
    action = (-carbon_scaled) / 5

    action = np.array([action], dtype=action_space.dtype)

    return action


def combine_policies(observation, action_space):
    day_night_action = battery_rbc_policy(observation, action_space).tolist()[0]
    grid_focused_action = grid_focused_policy(observation, action_space).tolist()[0]
    price_focused_action = price_focused_policy(observation, action_space).tolist()[0]
    carbon_focused_action = carbon_focused_policy(observation, action_space).tolist()[0]


    day_night_weight = 0.5
    grid_focused_weight = 0.01
    carbon_focused_weight = 0
    price_focused_weight = 0.05

    summed_weights = sum([day_night_weight, grid_focused_weight, price_focused_weight], carbon_focused_weight)
    if summed_weights > 0:
        weighted_average = ((day_night_weight * day_night_action)
                            + (grid_focused_weight * grid_focused_action)
                            + (price_focused_weight * price_focused_action)
                            + (carbon_focused_weight * carbon_focused_action)) / summed_weights
    else:
        weighted_average = 0
    action = np.array([weighted_average], dtype=action_space.dtype)

    return action


class MultiPolicyAgent:
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
        return combine_policies(observation, self.action_space[agent_id])
