import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class LiveLearner:

    def __init__(self, cap_learning_data, fit_delay_steps):
        self.cap_learning_data = cap_learning_data
        self.fit_delay_steps = fit_delay_steps

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

        print()
        if len(self.non_shiftable_loads) > 60 and len(self.non_shiftable_loads) % self.fit_delay_steps == 0:
            self.load_forecaster.fit(pd.Series(self.non_shiftable_loads))
            self.solar_forecaster.fit(pd.Series(self.solar_generations))

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
