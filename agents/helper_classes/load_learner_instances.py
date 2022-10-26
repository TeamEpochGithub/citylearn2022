import numpy as np
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import os.path as osp
import data.citylearn_challenge_2022_phase_1 as competition_data
from analysis import analysis_data


class LearnerInstance:

    def __init__(self, forecaster, lags, cap_learning_data, fit_delay_steps):
        self.trial_predictions = []
        self.trial_actual_values = []
        self.load_lags = lags
        self.max_load_lag = max(self.load_lags)
        self.cap_learning_data = cap_learning_data
        self.forecaster = forecaster
        self.non_shiftable_loads = []
        self.fit_delay_steps = fit_delay_steps
        self.fit_delay_counter = fit_delay_steps  # Since we want to fit at the beginning
        self.timestep = -1

        self.solar_generations = []

        carbon_intensities_path = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")
        carbon_intensities = pd.read_csv(carbon_intensities_path)

        weather_path = osp.join(osp.dirname(competition_data.__file__), "weather.csv")
        weather = pd.read_csv(weather_path)

        building_path = osp.join(osp.dirname(competition_data.__file__), "Building_1.csv")
        building = pd.read_csv(building_path)

        self.months = list(building["Month"])
        self.hours = list(building["Hour"])
        self.daytypes = list(building["Day Type"])

        self.temperatures = list(weather["Outdoor Drybulb Temperature [C]"])
        self.humidities = list(weather["Relative Humidity [%]"])

        self.temperatures_pred_6h = list(weather["6h Prediction Outdoor Drybulb Temperature [C]"])
        self.humidities_pred_6h = list(weather["6h Prediction Relative Humidity [%]"])

        self.carbon_intensities = list(carbon_intensities["kg_CO2/kWh"])

    def get_learner(self):
        return self.forecaster

    def get_lags(self):
        return self.load_lags

    def update_loads(self, load, model_is_used):
        self.timestep += 1

        if self.fit_delay_counter >= self.fit_delay_steps and len(self.non_shiftable_loads) > self.max_load_lag + 1:
            if model_is_used:
                self.fit_load()
                self.fit_delay_counter = 1
        else:
            self.fit_delay_counter += 1

        self.non_shiftable_loads.append(load)

        while len(self.non_shiftable_loads) > self.cap_learning_data:
            del self.non_shiftable_loads[0]

    def reset_trial(self):
        self.trial_predictions = []
        self.trial_actual_values = []

    def update_values(self, prediction, actual):
        self.trial_predictions.append(prediction)
        self.trial_actual_values.append(actual)

    def calculate_error(self):
        mse = 0
        for i in range(len(self.trial_predictions) - 1):
            mse += (self.trial_actual_values[i + 1] - self.trial_predictions[i]) ** 2

        mse = mse / (len(self.trial_predictions) - 1)
        return mse

    def fit_load(self):

        left_bound = max(self.timestep - self.cap_learning_data, 0)
        self.forecaster.fit(y=pd.Series(self.non_shiftable_loads),
                            exog=self.get_exogenuous_values(left_bound, self.timestep))

    def predict_load(self, steps):

        left_bound = self.timestep - self.max_load_lag
        right_bound = self.timestep + steps

        predictions = self.forecaster.predict(steps=steps,
                                              last_window=pd.Series(self.non_shiftable_loads[-self.max_load_lag:]),
                                              exog=self.get_exogenuous_values(left_bound, right_bound))

        if isinstance(predictions, pd.Series):
            predictions = np.asarray(predictions)
            predictions[predictions < 0] = 0
            return list(predictions)
        else:
            return [predictions]

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


class LoadLag0Learner(LearnerInstance):
    def __init__(self, cap_learning_data, fit_delay_steps):
        lags = [1, 2, 3, 4, 5, 23, 24, 25, 26, 27, 48, 49, 50, 51, 52]
        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=lags,
            transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        )
        super().__init__(forecaster, lags, cap_learning_data, fit_delay_steps)


class LoadLag1Learner(LearnerInstance):
    def __init__(self, cap_learning_data, fit_delay_steps):
        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                35, 36, 37, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=lags,
            transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        )
        super().__init__(forecaster, lags, cap_learning_data, fit_delay_steps)


class LoadLag2Learner(LearnerInstance):
    def __init__(self, cap_learning_data, fit_delay_steps):
        lags = [1, 2, 3, 4, 23, 24, 25, 26, 27, 47, 48, 49, 50, 51]
        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=lags,
            transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        )
        super().__init__(forecaster, lags, cap_learning_data, fit_delay_steps)


class LoadLag3Learner(LearnerInstance):
    def __init__(self, cap_learning_data, fit_delay_steps):
        lags = [1, 23, 45, 67]
        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=lags,
            transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        )
        super().__init__(forecaster, lags, cap_learning_data, fit_delay_steps)


class LoadLag4Learner(LearnerInstance):
    def __init__(self, cap_learning_data, fit_delay_steps):
        lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42, solver='sag', tol=0.001, alpha=3.593814),
            lags=lags,
            transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        )
        super().__init__(forecaster, lags, cap_learning_data, fit_delay_steps)


class LoadLag5Learner(LearnerInstance):
    def __init__(self, cap_learning_data, fit_delay_steps):
        lags = [1, 23, 45]
        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42, solver='sparse_cg', tol=0.1, alpha=12915.496650),
            lags=lags,
            transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        )
        super().__init__(forecaster, lags, cap_learning_data, fit_delay_steps)


class LoadLag6Learner(LearnerInstance):
    def __init__(self, cap_learning_data, fit_delay_steps):
        lags = [1, 23, 45, 67]
        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42, solver='sparse_cg', tol=0.1, alpha=12915.496650),
            lags=lags,
            transformer_y=FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        )
        super().__init__(forecaster, lags, cap_learning_data, fit_delay_steps)
