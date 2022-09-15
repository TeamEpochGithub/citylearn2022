import random
import numpy as np
import pandas as pd
import os
from bisect import bisect_left
import sys

# Split action space into discrete regions.
# Split environment space into discrete regions.

# Environment space: Electricity cost, Carbon cost, Hour, Total solar energy
# Array values: EC: 24, CC: 19, H: 2, TSE: 11 + 15,





class BasicQAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """
    def __init__(self):
        self.action_space = {}
        self.directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        self.q_tables = []
        self.pricing_intervals = [0.21, 0.29, 0.38, 0.46, 0.54]
        self.solar_intervals = [0, 478, 956, 1435, 1913]
        self.carbon_intervals = [0.07, 0.12, 0.18, 0.23, 0.28]
        self.actions = [-1.0, -0.78, -0.56, -0.33, -0.11, 0.11, 0.33, 0.56, 0.78, 1.0]

        self.learning_rate = 0.3
        self.discount_factor = 0.3
        self.exploration = 0.03

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def set_q_tables(self, agent_id):

        q_table_filename = f"p_c_s_size_5_space_10_agent_id_{agent_id}.csv"

        if q_table_filename not in os.listdir(f"{self.directory}/q_tables/"):
            print("Did not find:", agent_id)

        else:
            self.q_tables.append(pd.read_csv(f"{self.directory}/q_tables/{q_table_filename}", index_col=0))


    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        return self.q_policy(observation, agent_id)

    def q_policy(self, observation, agent_id):
        """
        Simple rule based policy based on day or night time
        """
        hour = observation[2]  # Hour index is 2 for all observations

        observed_price = observation[24]
        observed_carbon = observation[19]
        observed_solar = observation[11] + observation[15]

        observed_storage = observation[22]
        observed_consumption = observation[23]

        price = self.take_closest(self.pricing_intervals, observed_price)
        carbon = self.take_closest(self.carbon_intervals, observed_carbon)
        solar = self.take_closest(self.solar_intervals, observed_solar)

        q_table_row_name = f"p_{price}_c_{carbon}_s_{solar}"

        current_agent_table = self.q_tables[agent_id]

        q_table_row = list(current_agent_table.loc[q_table_row_name])
        max_q_value = max(q_table_row)

        chosen_action = None

        if self.decision(self.exploration) or max_q_value == 0:
            chosen_action = random.choice(self.actions)
        else:
            max_index = q_table_row.index(max_q_value)
            chosen_action = self.actions[max_index]

        reward = self.reward(price, carbon, chosen_action, observed_storage)

        previous_q = self.q_tables[agent_id].loc[q_table_row_name, str(chosen_action)]

        self.q_tables[agent_id].loc[q_table_row_name, str(chosen_action)] = previous_q + self.learning_rate * (reward + (self.discount_factor * max_q_value) - previous_q)

        action = chosen_action

        action = np.array([action], dtype=self.action_space[agent_id].dtype)
        assert self.action_space[agent_id].contains(action)
        return action

    def save_q_table(self, agent_id):

        self.q_tables[agent_id].to_csv(f"{self.directory}/q_tables/p_c_s_size_5_space_10_agent_id_{agent_id}.csv")

    # This function is from stackoverflow, and is authored by Lauritz V. Thaulow.
    def take_closest(self, myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.

        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return after
        else:
            return before

    def decision(self, probability):
        return random.random() < probability

    def reward(self, price, carbon, action, storage):

        return (price + carbon + storage) * -1 * action

    # from main.QLearning import QLearning
    #
    # class MyQLearning(QLearning):
    #
    #     def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
    #         # The Q value from the current state that will be used as the old value when the Q is updated.
    #         previous_q = self.get_q(state, action)
    #
    #         # The Q value belonging to the best possible action for the next state.
    #         max_q = max(self.get_action_values(state_next, possible_actions))
    #
    #         # Implementation of the update rule.
    #         new_q = previous_q + alfa * (r + (gamma * max_q) - previous_q)
    #
    #         # Setting the newly computed Q value.
    #         self.set_q(state, action, new_q)
    #
    #         return

