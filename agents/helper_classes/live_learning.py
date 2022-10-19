import sys

import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


class LiveLearner:

    def __init__(self, cap_learning_data):
        self.cap_learning_data = cap_learning_data

        self.load_forecaster = ForecasterAutoreg(
            regressor=RandomForestRegressor(random_state=42),
            lags=[1, 2, 25, 26],
            transformer_y=StandardScaler()
        )

        self.solar_forecaster = ForecasterAutoreg(
            regressor=RandomForestRegressor(random_state=42),
            lags=[1, 2, 25, 26],
            transformer_y=StandardScaler()
        )

        self.non_shiftable_loads = []
        self.solar_generations = []

    def update_lists(self, observation):
        non_shiftable_load = observation[20]
        scaled_non_shiftable_load = non_shiftable_load / 8
        solar_generation = observation[21]
        scaled_solar_generation = solar_generation / 4

        self.non_shiftable_loads.append(scaled_non_shiftable_load)
        self.solar_generations.append(scaled_solar_generation)

        if len(self.non_shiftable_loads) > self.cap_learning_data:
            del self.non_shiftable_loads[0]
            del self.solar_generations[0]

    def fit_and_predict_load(self, steps):
        self.load_forecaster.fit(pd.Series(self.non_shiftable_loads))
        predicted_load = self.load_forecaster.predict(steps=steps).iloc[steps - 1]
        if predicted_load < 0:
            predicted_load = 0
        return predicted_load * 8

    def fit_and_predict_solar(self, steps):
        self.solar_forecaster.fit(pd.Series(self.solar_generations))
        predicted_solar = self.solar_forecaster.predict(steps=steps).iloc[steps - 1]
        if predicted_solar < 0:
            predicted_solar = 0
        return predicted_solar * 4

    def predict_consumption(self, steps):
        return self.fit_and_predict_load(steps) - self.fit_and_predict_solar(steps)

    def fit_and_predict_multiple_load(self, steps, fit=False):
        if fit:
            self.load_forecaster.fit(pd.Series(self.non_shiftable_loads))
        predicted_load = self.load_forecaster.predict(steps=steps)
        for i, x in enumerate(predicted_load):
            if x < 0:
                predicted_load[i] = 0
        # print("load", predicted_load)
        # print("load over")
        return list(predicted_load * 8)[:steps]

    def fit_and_predict_multiple_solar(self, steps, fit=False):
        if fit:
            self.solar_forecaster.fit(pd.Series(self.solar_generations))
        predicted_solar = self.solar_forecaster.predict(steps=steps)
        for i, x in enumerate(predicted_solar):
            if x < 0:
                predicted_solar[i] = 0
        return list(predicted_solar * 4)[:steps]

    def predict_multiple_consumption(self, steps, fit=False):
        return [a - b for a, b in
                zip(self.fit_and_predict_multiple_load(steps, fit), self.fit_and_predict_multiple_solar(steps, fit))]
