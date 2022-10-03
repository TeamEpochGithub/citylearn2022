import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as mp

#Do some stuff
# price_df = pd.read_csv("../data/citylearn_challenge_2022_phase_1/pricing.csv")
# carbon_df = pd.read_csv("../data/citylearn_challenge_2022_phase_1/carbon_intensity.csv")
#
# price_data = price_df["Electricity Pricing [$]"]
# max_price = price_data.max()
# min_price = price_data.min()
#
# print("Price", max_price, min_price)
#
# carbon_data = carbon_df["kg_CO2/kWh"]
# max_carbon = carbon_data.max()
# min_carbon = carbon_data.min()
# print("Carbon: ", max_carbon, min_carbon)
#
# X = pd.DataFrame({'Electricity Pricing [$]': price_data, "kg_CO2/kWh": carbon_data})
#
# X["cost"] = X['Electricity Pricing [$]'] + X["kg_CO2/kWh"]
# max_cost = X["cost"].max()
# min_cost = X["cost"].min()
# mean_cost = X["cost"].mean()
#
# print("Cost", max_cost, min_cost, mean_cost)
#
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
# x_scaled = min_max_scaler.fit_transform(X)
# df = pd.DataFrame(x_scaled)
#
# print(df[2].mean())
# print(df[2].median())
# print(df[2].sum())
# print(df[2].max())
#
# df.iloc[:50].plot(y=2, use_index=True)
#
# mp.show()
#
#
# print(df)

def policy_cost_battery_mean(observation, action_space):

    price = observation[24]
    carbon = observation[19]
    battery = observation[22]

    max_cost = 0.8217962171196289
    min_cost = 0.2803828731473388
    mean_cost_normalized = -0.4485576576977546

    cost = price + carbon
    cost_std = (cost - min_cost) / (max_cost - min_cost)
    cost_scaled = cost_std * 2 - 1

    action = -cost_scaled /4

    if cost_scaled < mean_cost_normalized:
        action = -cost_scaled / 3

    if cost_scaled > mean_cost_normalized and cost_scaled < 0:
        action = 0

    action = np.array([action], dtype=action_space.dtype)

    return action


def policy_cost_battery_minmax(observation, action_space):
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    max_cost = 0.8217962171196289
    min_cost = 0.2803828731473388

    price = observation[24]
    carbon = observation[19]
    battery = observation[22]

    ruimte = 1-battery

    cost = price + carbon
    cost_std = (cost - min_cost) / (max_cost - min_cost)
    cost_scaled = cost_std * 2 - 1

    action = -cost_scaled / 3

    if cost_scaled < 1 and ruimte < -cost_scaled:
        action = ruimte / 2

    if cost_scaled > 1 and ruimte < cost_scaled:
        action = ruimte

    # action = -cost_scaled / 3

    action = np.array([action], dtype=action_space.dtype)

    return action

class Test:
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
        return policy_cost_battery_minmax(observation, self.action_space[agent_id])
