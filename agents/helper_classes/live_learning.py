import csv

import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os.path as osp
import data.citylearn_challenge_2022_phase_1 as competition_data
from analysis import analysis_data


def write_prediction_to_file(load, solar):
    row = [load, solar]
    action_file_path = osp.join(osp.dirname(analysis_data.__file__), 'pred_consumption_predictions.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()


class LiveLearner:

    def __init__(self, cap_learning_data, fit_delay_steps, write_to_file):
        self.cap_learning_data = cap_learning_data
        self.fit_delay_steps = fit_delay_steps
        self.fit_delay_counter = self.fit_delay_steps
        self.write_to_file = write_to_file
        self.timestep = -1

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

        carbon_intensities_path = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")
        carbon_intensities = pd.read_csv(carbon_intensities_path)

        weather_path = osp.join(osp.dirname(competition_data.__file__), "weather.csv")
        weather = pd.read_csv(weather_path)

        building_path = osp.join(osp.dirname(competition_data.__file__), "Building_1.csv")
        building = pd.read_csv(building_path)

        self.non_shiftable_loads = []
        self.solar_generations = []

        self.months = list(building["Month"])
        self.hours = list(building["Hour"])
        self.daytypes = list(building["Day Type"])

        self.temperatures = list(weather["Outdoor Drybulb Temperature [C]"])
        self.humidities = list(weather["Relative Humidity [%]"])

        self.temperatures_pred_6h = list(weather["6h Prediction Outdoor Drybulb Temperature [C]"])
        self.humidities_pred_6h = list(weather["6h Prediction Relative Humidity [%]"])

        self.carbon_intensities = list(carbon_intensities["kg_CO2/kWh"])

    def update_lists(self, observation):
        self.timestep += 1
        if self.fit_delay_counter >= self.fit_delay_steps and \
                len(self.non_shiftable_loads) > self.max_load_lag + 1 and \
                len(self.solar_generations) > self.max_solar_lag + 1:
            self.force_fit()
            self.fit_delay_counter = 0
        else:
            self.fit_delay_counter += 1

        self.non_shiftable_loads.append(observation[20])
        self.solar_generations.append(observation[21])

        if len(self.non_shiftable_loads) > self.cap_learning_data:
            del self.non_shiftable_loads[0]
            del self.solar_generations[0]

    def force_fit_load(self):

        left_bound = max(self.timestep - self.cap_learning_data, 0)
        self.load_forecaster.fit(y=pd.Series(self.non_shiftable_loads),
                                 exog=self.get_exogenuous_values(left_bound, self.timestep))

    def force_fit_solar(self):
        self.solar_forecaster.fit(pd.Series(self.solar_generations))

    def force_fit(self):
        self.force_fit_load()
        self.force_fit_solar()

    def get_exogenuous_values(self, left_bound, right_bound):
        return pd.concat([
                          pd.Series(self.months[left_bound:right_bound]),
                          # pd.Series(self.hours[left_bound:right_bound]),
                          # pd.Series(self.daytypes[left_bound:right_bound]),
                          pd.Series(self.temperatures[left_bound:right_bound]),
                          pd.Series(self.humidities[left_bound:right_bound]),
                          pd.Series(self.temperatures_pred_6h[left_bound:right_bound]),
                          pd.Series(self.humidities_pred_6h[left_bound:right_bound]),
                          pd.Series(self.carbon_intensities[left_bound:right_bound])],
                         axis=1)

    def predict_non_shiftable_load(self, steps):

        left_bound = self.timestep - self.max_load_lag
        right_bound = self.timestep + steps

        predictions = self.load_forecaster.predict(steps=steps,
                                                   last_window=pd.Series(self.non_shiftable_loads[-self.max_load_lag:]),
                                                   exog=self.get_exogenuous_values(left_bound, right_bound))

        for i, x in enumerate(predictions):
            if x < 0:
                predictions[i] = 0

        return list(predictions)

    def predict_solar_generations(self, steps):

        predictions = self.solar_forecaster.predict(steps=steps,
                                                    last_window=pd.Series(self.solar_generations[-self.max_solar_lag:]))

        for i, x in enumerate(predictions):
            if x < 0:
                predictions[i] = 0

        return list(predictions)

    def predict_consumption(self, steps, only_write_once):

        load = self.predict_non_shiftable_load(steps)
        solar = self.predict_solar_generations(steps)

        if self.write_to_file:
            if only_write_once:
                write_prediction_to_file(load[0], solar[0])

        return [a - b for a, b in
                zip(load, solar)]
