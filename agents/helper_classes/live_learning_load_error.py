import csv

import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os.path as osp
import data.citylearn_challenge_2022_phase_1 as competition_data
from analysis import analysis_data


def write_prediction_to_file(agent_id, timestep, load_error):
    row = [agent_id, timestep, load_error]
    action_file_path = osp.join(osp.dirname(analysis_data.__file__), 'pred_load_error_predictions.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()


class LiveLearnerLoadError:

    def __init__(self, cap_learning_data, fit_delay_steps, write_to_file, agent_id):
        self.cap_learning_data = cap_learning_data
        self.fit_delay_steps = fit_delay_steps
        self.fit_delay_counter = self.fit_delay_steps
        self.write_to_file = write_to_file
        self.timestep = -1
        self.agent_id = agent_id

        self.load_error_lags = [1, 2, 3, 4, 5, 23, 24, 25, 26, 27, 48, 49, 50, 51, 52]

        self.max_load_error_lag = max(self.load_error_lags)

        self.load_error_forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=self.load_error_lags,
            transformer_y=StandardScaler()
        )

        carbon_intensities_path = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")
        carbon_intensities = pd.read_csv(carbon_intensities_path)

        weather_path = osp.join(osp.dirname(competition_data.__file__), "weather.csv")
        weather = pd.read_csv(weather_path)

        building_path = osp.join(osp.dirname(competition_data.__file__), "Building_1.csv")
        building = pd.read_csv(building_path)

        self.non_shiftable_loads = []
        self.load_predictions = []
        self.load_errors = []

        self.months = list(building["Month"])
        self.hours = list(building["Hour"])
        self.daytypes = list(building["Day Type"])

        self.temperatures = list(weather["Outdoor Drybulb Temperature [C]"])
        self.humidities = list(weather["Relative Humidity [%]"])

        self.temperatures_pred_6h = list(weather["6h Prediction Outdoor Drybulb Temperature [C]"])
        self.humidities_pred_6h = list(weather["6h Prediction Relative Humidity [%]"])

        self.carbon_intensities = list(carbon_intensities["kg_CO2/kWh"])

    def update_non_shiftable_loads(self, load):
        self.timestep += 1
        self.non_shiftable_loads.append(load)

        if len(self.non_shiftable_loads) > 3:
            del self.non_shiftable_loads[0]

    def update_load_predictions(self, load):
        self.load_predictions.append(load)

        if len(self.load_predictions) > 3:
            del self.load_predictions[0]

        if self.timestep > 125:
            self.update_load_errors()

    def update_load_errors(self):
        if self.fit_delay_counter >= self.fit_delay_steps and \
                len(self.load_errors) > self.max_load_error_lag + 1:
            self.force_fit_load()
            self.fit_delay_counter = 0
        else:
            self.fit_delay_counter += 1
        non_shiftable_load = self.non_shiftable_loads[-1]
        load_prediction = self.load_predictions[-2]

        self.load_errors.append(non_shiftable_load - load_prediction)

        if len(self.load_errors) > self.cap_learning_data:
            del self.load_errors[0]

    def force_fit_load(self):
        left_bound = max(self.timestep - len(self.load_errors), 0)
        self.load_error_forecaster.fit(y=pd.Series(self.load_errors),
                                       exog=self.get_exogenuous_values(left_bound, self.timestep))

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

    def predict_load_error(self, steps):

        left_bound = self.timestep - self.max_load_error_lag
        right_bound = self.timestep + steps

        predictions = self.load_error_forecaster.predict(steps=steps,
                                                         last_window=pd.Series(
                                                             self.load_errors[-self.max_load_error_lag:]),
                                                         exog=self.get_exogenuous_values(left_bound, right_bound))

        for i, x in enumerate(predictions):
            if x < 0:
                predictions[i] = 0

        if self.write_to_file:
            write_prediction_to_file(self.agent_id, self.timestep, predictions[0])

        return list(predictions)
