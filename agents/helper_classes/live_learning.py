import sys

import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


class LiveLearner:

    def __init__(self, cap_learning_data, fit_delay_steps):
        self.cap_learning_data = cap_learning_data
        self.fit_delay_steps = fit_delay_steps

        self.load_forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=[1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30],
            transformer_y=StandardScaler()
        )

        self.solar_forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=[1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30],
            transformer_y=StandardScaler()
        )

        self.non_shiftable_loads = []
        self.solar_generations = []

        self.fit_delay_non_shiftable_load_buffer = []
        self.fit_delay_solar_generations_buffer = []

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

        # if len(self.non_shiftable_loads) > 60 and len(self.non_shiftable_loads) % self.fit_delay_steps == 0:
        #     self.load_forecaster.fit(pd.Series(self.non_shiftable_loads))
        #     self.solar_forecaster.fit(pd.Series(self.solar_generations))

    def force_fit_load(self):
        self.load_forecaster.fit(pd.Series(self.non_shiftable_loads))
        # print("Fitting load")

    def force_fit_solar(self):
        self.solar_forecaster.fit(pd.Series(self.solar_generations))

    def force_fit(self):
        self.force_fit_load()
        self.force_fit_solar()
        self.fit_delay_non_shiftable_load_buffer = []
        self.fit_delay_solar_generations_buffer = []

    def predict_non_shiftable_load(self, steps):

        if len(self.fit_delay_non_shiftable_load_buffer) == 0 or len(self.fit_delay_non_shiftable_load_buffer) < steps:

            self.force_fit_load()

            self.fit_delay_non_shiftable_load_buffer = self.predict_multiple_loads(steps + self.fit_delay_steps)

        predictions = self.fit_delay_non_shiftable_load_buffer[:steps]
        del self.fit_delay_non_shiftable_load_buffer[0]

        return predictions

    def predict_solar_generations(self, steps):

        if len(self.fit_delay_solar_generations_buffer) == 0 or len(self.fit_delay_solar_generations_buffer) < steps:

            self.force_fit_solar()

            self.fit_delay_solar_generations_buffer = self.predict_multiple_solars(steps + self.fit_delay_steps)

        predictions = self.fit_delay_solar_generations_buffer[:steps]
        del self.fit_delay_solar_generations_buffer[0]

        return predictions

    def fit_delay_buffer_consumption(self, steps):

        # print("Predicting")

        load = self.predict_non_shiftable_load(steps)
        solar = self.predict_solar_generations(steps)

        return [a - b for a, b in
                zip(load, solar)]


    def predict_load(self, steps):
        predicted_load = self.load_forecaster.predict(steps=steps).iloc[steps - 1]
        if predicted_load < 0:
            predicted_load = 0
        return predicted_load * 8

    def predict_solar(self, steps):
        predicted_solar = self.solar_forecaster.predict(steps=steps).iloc[steps - 1]
        if predicted_solar < 0:
            predicted_solar = 0
        return predicted_solar * 4

    def predict_consumption(self, steps):
        return self.predict_load(steps) - self.predict_solar(steps)

    def predict_multiple_loads(self, steps):
        predicted_load = self.load_forecaster.predict(steps=steps)
        for i, x in enumerate(predicted_load):
            if x < 0:
                predicted_load[i] = 0
        return list(predicted_load * 8)[:steps]

    def predict_multiple_solars(self, steps):
        predicted_solar = self.solar_forecaster.predict(steps=steps)
        for i, x in enumerate(predicted_solar):
            if x < 0:
                predicted_solar[i] = 0
        return list(predicted_solar * 4)[:steps]

    def predict_multiple_consumptions(self, steps):
        if self.fit_delay_steps > 1:
            self.load_forecaster.fit(pd.Series(self.non_shiftable_loads))
            self.solar_forecaster.fit(pd.Series(self.solar_generations))
        return [a - b for a, b in
                zip(self.predict_multiple_loads(steps), self.predict_multiple_solars(steps))]
