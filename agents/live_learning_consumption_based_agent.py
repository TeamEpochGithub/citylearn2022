import numpy as np
import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge

from sklearn.preprocessing import StandardScaler


class LiveLearningAgent:

    def __init__(self, action_space, cap_learning_data, agent_id):

        self.action_space = action_space
        self.timestep = -1
        self.cap_learning_data = cap_learning_data
        self.agent_id = agent_id
        self.all_actions = []

        self.load_forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=[1, 2, 3, 4, 5, 23, 24, 25, 26, 27, 48, 49, 50, 51, 52],
            transformer_y=StandardScaler()
        )

        self.solar_forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=[1, 2, 3, 23, 24, 25, 48, 49, 50],
            transformer_y=StandardScaler()
        )

        self.non_shiftable_loads = []
        self.solar_generations = []

    def compute_action(self, observation):
        """Get observation return action"""

        self.timestep += 1
        hour = observation[2]
        electrical_storage_soc = observation[22]
        net_electricity_consumption = observation[23]

        non_shiftable_load = observation[20]
        scaled_non_shiftable_load = non_shiftable_load / 8
        solar_generation = observation[21]
        scaled_solar_generation = solar_generation / 4

        self.non_shiftable_loads.append(scaled_non_shiftable_load)
        self.solar_generations.append(scaled_solar_generation)

        if len(self.non_shiftable_loads) > self.cap_learning_data:
            del self.non_shiftable_loads[0]
            del self.solar_generations[0]

        if self.timestep >= 60:
            predicted_load = self.fit_and_predict_load() * 8
            predicted_solar = self.fit_and_predict_solar() * 4
            if predicted_solar < 0:
                predicted_solar = 0
            if predicted_load < 0:
                predicted_load = 0

            action = -((predicted_load - predicted_solar) / 6.4)
            action = action - np.mean(self.all_actions)

            # action_scaler = 1
            # if predicted_load < predicted_solar:
            #     action_scaler = 1.2

            # action = action * action_scaler
            if self.agent_id == 1:
                # print(non_shiftable_load, solar_generation, "D:", self.timestep//24, " h:", hour)
                # print(predicted_load, predicted_solar, "D:", self.timestep//24, " h:", hour)
                # print("D:", self.timestep // 24, " h:", hour, " load: ", predicted_load, " solar: ", predicted_solar,
                #       " action: ", action, " storage: ", electrical_storage_soc, " consumption: ",
                #       net_electricity_consumption)
                print(self.timestep, action)
        else:

            action = -0.067
            if 6 <= hour <= 14:
                action = 0.11

        self.all_actions.append(action)
        action = np.array([action], dtype=self.action_space.dtype)
        return action

    def fit_and_predict_load(self):

        self.load_forecaster.fit(pd.Series(self.non_shiftable_loads))
        predicted_load = self.load_forecaster.predict(steps=1).iloc[0]

        return predicted_load

    def fit_and_predict_solar(self):

        self.solar_forecaster.fit(pd.Series(self.solar_generations))
        predicted_solar = self.solar_forecaster.predict(steps=1).iloc[0]

        return predicted_solar


class LiveLearningAgentBuilder:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}
        self.agents = {}
        self.cap_learning_data = 1500

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        if str(agent_id) not in self.agents:
            self.agents[str(agent_id)] = LiveLearningAgent(self.action_space[agent_id], self.cap_learning_data,
                                                           agent_id)

        return self.agents[str(agent_id)].compute_action(observation[agent_id])
