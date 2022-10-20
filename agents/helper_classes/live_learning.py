import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class LiveLearner:

    def __init__(self, cap_learning_data, fit_delay_steps):
        self.cap_learning_data = cap_learning_data
        self.fit_delay_steps = fit_delay_steps
        self.fit_delay_counter = self.fit_delay_steps

        self.load_lags = [1, 2, 3, 4, 5, 23, 24, 25, 26, 27, 48, 49, 50, 51, 52]
        self.solar_lags = [1, 2, 3, 23, 24, 25, 48, 49, 50]

        self.max_load_lag = max(self.load_lags)
        self.max_solar_lag = max(self.solar_lags)

        self.load_forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=self.load_lags,
            transformer_y=StandardScaler()
        )

        self.solar_forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=self.solar_lags,
            transformer_y=StandardScaler()
        )

        self.non_shiftable_loads = []
        self.solar_generations = []


    def update_lists(self, observation):
        non_shiftable_load = observation[20]
        scaled_non_shiftable_load = non_shiftable_load / 8
        solar_generation = observation[21]
        scaled_solar_generation = solar_generation / 4

        if self.fit_delay_counter >= self.fit_delay_steps and\
                len(self.non_shiftable_loads) > self.max_load_lag + 1 and\
                len(self.solar_generations) > self.max_solar_lag + 1:
            self.force_fit()
            self.fit_delay_counter = 0
        else:
            self.fit_delay_counter += 1

        self.non_shiftable_loads.append(scaled_non_shiftable_load)
        self.solar_generations.append(scaled_solar_generation)

        if len(self.non_shiftable_loads) > self.cap_learning_data:
            del self.non_shiftable_loads[0]
            del self.solar_generations[0]

    def force_fit_load(self):
        self.load_forecaster.fit(pd.Series(self.non_shiftable_loads))

    def force_fit_solar(self):
        self.solar_forecaster.fit(pd.Series(self.solar_generations))

    def force_fit(self):
        self.force_fit_load()
        self.force_fit_solar()

    def predict_non_shiftable_load(self, steps):

        predictions = self.load_forecaster.predict(steps=steps,
                                                   last_window=pd.Series(self.non_shiftable_loads[-self.max_load_lag:]))

        for i, x in enumerate(predictions):
            if x < 0:
                predictions[i] = 0

        return list(predictions * 8)

    def predict_solar_generations(self, steps):

        predictions = self.solar_forecaster.predict(steps=steps,
                                                    last_window=pd.Series(self.solar_generations[-self.max_solar_lag:]))

        for i, x in enumerate(predictions):
            if x < 0:
                predictions[i] = 0

        return list(predictions * 4)

    def predict_consumption(self, steps):

        load = self.predict_non_shiftable_load(steps)
        solar = self.predict_solar_generations(steps)

        return [a - b for a, b in
                zip(load, solar)]
