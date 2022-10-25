import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error

import os.path as osp
from data import citylearn_challenge_2022_phase_1 as competition_data

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiSeries import ForecasterAutoregMultiSeries

from skranger.ensemble import RangerForestRegressor

from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

from prophet import Prophet
import datetime

import logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


def fix_data():
    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\consumption_prediction\load_data.csv")

    count = 8758

    for i in range(1, 5):
        hour_23 = load_df.iloc[(i * count) - 1]
        print(hour_23)
        hour_23["hour"] = 23
        print(hour_23)

        hour_24 = load_df.iloc[(i * count) - 1]
        print(hour_24)
        hour_24["hour"] = 24
        print(hour_24)

        load_df = pd.concat([load_df.iloc[:(i * count)], hour_23, hour_24, load_df.iloc[(i * count):]])

    load_df.to_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\consumption_prediction\load_data_fixed.csv",
        index=False)


def plot_and_save(title, xs, ys, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.savefig(title)
    plt.show()


def plot_average_lf():
    building_1_data = osp.join(osp.dirname(competition_data.__file__), "building_1.csv")
    building_2_data = osp.join(osp.dirname(competition_data.__file__), "building_2.csv")
    building_3_data = osp.join(osp.dirname(competition_data.__file__), "building_3.csv")
    building_4_data = osp.join(osp.dirname(competition_data.__file__), "building_4.csv")
    building_5_data = osp.join(osp.dirname(competition_data.__file__), "building_5.csv")

    take_first = 8760

    b1_df = pd.read_csv(building_1_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b2_df = pd.read_csv(building_2_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b3_df = pd.read_csv(building_3_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b4_df = pd.read_csv(building_4_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b5_df = pd.read_csv(building_5_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]

    x_axis = np.arange(0, 8760)[0:24]

    days = 365

    b1_df = np.split(b1_df, days)
    b2_df = np.split(b2_df, days)
    b3_df = np.split(b3_df, days)
    b4_df = np.split(b4_df, days)
    b5_df = np.split(b5_df, days)

    b1_df = np.asarray(b1_df).mean(axis=0)
    b2_df = np.asarray(b2_df).mean(axis=0)
    b3_df = np.asarray(b3_df).mean(axis=0)
    b4_df = np.asarray(b4_df).mean(axis=0)
    b5_df = np.asarray(b5_df).mean(axis=0)

    plt.plot(x_axis, b1_df)
    plt.plot(x_axis, b2_df)
    plt.plot(x_axis, b3_df)
    plt.plot(x_axis, b4_df)
    plt.plot(x_axis, b5_df)

    plt.show()


def day_difference():
    building_1_data = osp.join(osp.dirname(competition_data.__file__), "building_1.csv")
    building_2_data = osp.join(osp.dirname(competition_data.__file__), "building_2.csv")
    building_3_data = osp.join(osp.dirname(competition_data.__file__), "building_3.csv")
    building_4_data = osp.join(osp.dirname(competition_data.__file__), "building_4.csv")
    building_5_data = osp.join(osp.dirname(competition_data.__file__), "building_5.csv")

    take_first = 8760

    b1_df = pd.read_csv(building_1_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b2_df = pd.read_csv(building_2_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b3_df = pd.read_csv(building_3_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b4_df = pd.read_csv(building_4_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b5_df = pd.read_csv(building_5_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]

    arr_list = [b1_df, b2_df, b3_df, b4_df, b5_df]
    diff_list = [[], [], [], [], []]

    window_size = 168

    for i in range(window_size, take_first):

        for j in range(0, 5):
            diff_list[j].append(np.square(arr_list[j][i] - arr_list[j][i - window_size]))

    print(np.mean(diff_list[0]))
    print(np.mean(diff_list[1]))
    print(np.mean(diff_list[2]))
    print(np.mean(diff_list[3]))
    print(np.mean(diff_list[4]))


def average_difference():
    building_1_data = osp.join(osp.dirname(competition_data.__file__), "building_1.csv")
    building_2_data = osp.join(osp.dirname(competition_data.__file__), "building_2.csv")
    building_3_data = osp.join(osp.dirname(competition_data.__file__), "building_3.csv")
    building_4_data = osp.join(osp.dirname(competition_data.__file__), "building_4.csv")
    building_5_data = osp.join(osp.dirname(competition_data.__file__), "building_5.csv")

    take_first = 8760

    b1_df = pd.read_csv(building_1_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b2_df = pd.read_csv(building_2_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b3_df = pd.read_csv(building_3_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b4_df = pd.read_csv(building_4_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b5_df = pd.read_csv(building_5_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]

    x_axis = np.arange(0, 8760)[0:24]

    days = 365

    b1_df_mean = np.split(b1_df, days)
    b2_df_mean = np.split(b2_df, days)
    b3_df_mean = np.split(b3_df, days)
    b4_df_mean = np.split(b4_df, days)
    b5_df_mean = np.split(b5_df, days)

    b1_df_mean = np.asarray(b1_df_mean).mean(axis=0)
    b2_df_mean = np.asarray(b2_df_mean).mean(axis=0)
    b3_df_mean = np.asarray(b3_df_mean).mean(axis=0)
    b4_df_mean = np.asarray(b4_df_mean).mean(axis=0)
    b5_df_mean = np.asarray(b5_df_mean).mean(axis=0)

    b1_df_mean = np.concatenate([b1_df_mean] * days, axis=0)
    b2_df_mean = np.concatenate([b2_df_mean] * days, axis=0)
    b3_df_mean = np.concatenate([b3_df_mean] * days, axis=0)
    b4_df_mean = np.concatenate([b4_df_mean] * days, axis=0)
    b5_df_mean = np.concatenate([b5_df_mean] * days, axis=0)

    error_list = []

    for i in range(1, take_first):
        past_mean = b1_df_mean[i - 1]
        current_mean = b1_df_mean[i]

        factor = current_mean / past_mean

        past_load = b1_df[i - 1]

        predicted_load = past_load * factor

        current_load = b1_df[i]

        error_list.append(np.square(predicted_load - current_load))

    print(np.asarray(error_list).mean())

    sys.exit()

    mse_b1 = (np.square(b1_df_mean - b1_df)).mean()
    mse_b2 = (np.square(b2_df_mean - b2_df)).mean()
    mse_b3 = (np.square(b3_df_mean - b3_df)).mean()
    mse_b4 = (np.square(b4_df_mean - b4_df)).mean()
    mse_b5 = (np.square(b5_df_mean - b5_df)).mean()

    print(mse_b1)
    print(mse_b2)
    print(mse_b3)
    print(mse_b4)
    print(mse_b5)


def auto_regressor_test_non_grid():
    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv")

    load_df[load_df.columns] = MinMaxScaler().fit_transform(load_df[load_df.columns])

    load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
    load_df_target = load_df["non_shiftable_load_future"]

    total_records = 8758

    test_fraction = 0.2
    test_count = round(8758 * test_fraction)

    validation_fraction = 0.7
    validation_count = round(8758 * validation_fraction)

    metric_list = []
    non_backtest_metric = []

    for index, data in enumerate(np.split(load_df_target, 5)):

        print(f"------House {index + 1}-----")

        data_train_val = data.iloc[:validation_count]
        data_train_test = data.iloc[:test_count]

        encountered_data = list(data_train_test)

        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=[1,2,3,4,5,23,24,25,26,27, 48, 49, 50, 51, 52],
            transformer_y=StandardScaler()
        )

        forecaster.fit(pd.Series(data_train_test.values))

        predictions = []
        truth = []

        for index, current in enumerate(data.iloc[test_count:]):

            if index % 200 == 0:
                print(index)

            # metric, predictions_train = backtesting_forecaster(
            #     forecaster=forecaster,
            #     y=pd.Series(encountered_data),
            #     initial_train_size=len(encountered_data)-1,
            #     steps=1,
            #     metric='mean_squared_error',
            #     refit=False,
            #     verbose=False
            # )
            # predictions.append(np.asarray(predictions_train)[-1][0])

            current_prediction = forecaster.predict(steps=1)

            predictions.append(current_prediction.iloc[0])

            truth.append(current)

            encountered_data.append(current)

            if len(encountered_data) > 1500:
                del encountered_data[0]

            if index % 2 == 0:
                forecaster.fit(pd.Series(encountered_data))

        mse = (np.square(np.asarray(predictions) - np.asarray(truth))).mean()
        non_backtest_metric.append(np.sqrt(mse))

        print("Custom mse:", mse)

        metric, predictions_backtest = backtesting_forecaster(
            forecaster=forecaster,
            y=data,
            initial_train_size=len(data_train_test),
            fixed_train_size=False,
            steps=24,
            metric="mean_squared_error",
            refit=False,
            verbose=False
        )

        metric_list.append(metric)


        print(f'Backtest error: {metric}')


        fig, ax = plt.subplots(figsize=(12, 3.5))
        pd.Series(truth).plot(ax=ax, linewidth=2, label='real')
        pd.Series(np.asarray(predictions)).plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title(f'Prediction vs real non shiftable load of house {index+1}')
        ax.legend()

        plt.show()

    print(metric_list)
    print(non_backtest_metric)


def auto_regressor_test_grid():
    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv")

    load_df[load_df.columns] = MinMaxScaler().fit_transform(load_df[load_df.columns])

    load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
    load_df_target = load_df["non_shiftable_load_future"]

    total_records = 8758

    test_fraction = 0.90
    test_count = round(8758 * test_fraction)

    validation_fraction = 0.7
    validation_count = round(8758 * validation_fraction)

    metric_list = []
    grid_list = []

    for index, data in enumerate(np.split(load_df_target, 5)):
        data_train_val = data.iloc[:validation_count]
        data_train_test = data.iloc[:test_count]

        forecaster = ForecasterAutoreg(
            regressor=XGBRegressor(),
            lags=[1,2,3,23,24,25, 48, 49, 50],
            transformer_y=StandardScaler()
        )

        lags_grid = [[1,2,3,4,5], [1,2,3,23,24,25], [1,2,3,4,5,23,24,25,26,27], 5, 10, 15, 20, 25, [1,2,3,23,24,25, 48, 49, 50], [1,2,3,4,5,23,24,25,26,27, 48, 49, 50, 51, 52]]

        param_grid = {'alpha': np.logspace(-3, 5, 10)}

        results_grid = grid_search_forecaster(
            forecaster=forecaster,
            y=data,
            param_grid=param_grid,
            lags_grid=lags_grid,
            steps=24,
            metric='mean_squared_error',
            refit=False,
            initial_train_size=len(data_train_val),
            fixed_train_size=False,
            return_best=True,
            verbose=False
        )

        # forecaster.fit(data_train_test)
        # #
        metric, predictions = backtesting_forecaster(
            forecaster=forecaster,
            y=data,
            initial_train_size=len(data_train_test),
            fixed_train_size=False,
            steps=24,
            metric="mean_squared_error",
            refit=False,
            verbose=False
        )

        metric_list.append(metric)
        # grid_list.append(results_grid)

        print(f"------House {index+1}-----")
        print(f'Backtest error: {metric}')

        fig, ax = plt.subplots(figsize=(12, 3.5))
        data.loc[predictions.index].plot(ax=ax, linewidth=2, label='real')
        predictions.plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title(f'Prediction vs real non shiftable load of house {index+1}')
        ax.legend()

        plt.show()

    print(metric_list)


def auto_regressor_test_generic_start_exo():
    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv")

    load_df[load_df.columns] = MinMaxScaler().fit_transform(load_df[load_df.columns])

    load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
    load_df_target = load_df["non_shiftable_load_future"]
    hours_df = load_df["hour"]
    hours_df_split = np.split(hours_df, 5)

    total_records = 8758

    test_fraction = 0.2
    test_count = round(8758 * test_fraction)

    validation_fraction = 0.7
    validation_count = round(8758 * validation_fraction)

    metric_list = []
    non_backtest_metric = []

    for index, data in enumerate(np.split(load_df_target, 5)):

        print(f"------House {index + 1}-----")

        data_train_val = data.iloc[:validation_count]
        data_train_test = data.iloc[:test_count]

        lag_size = 60

        data_minimal = data.iloc[:lag_size]
        hours_minimal = hours_df_split[index].iloc[:lag_size]

        encountered_data = list(data_minimal)
        encountered_hours = list(hours_minimal)

        future_hours = list(hours_df_split[index].iloc[lag_size:])

        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=[1,2,3,4,5,23,24,25,26,27, 48, 49, 50, 51, 52],
            transformer_y=StandardScaler()
        )

        forecaster.fit(pd.Series(data_minimal.values), exog=pd.Series(encountered_hours))

        predictions = []
        truth = []

        for inner_index, current in enumerate(data.iloc[lag_size:]):

            if inner_index % 200 == 0:
                print(inner_index)

            current_prediction = forecaster.predict(steps=1, exog=pd.Series(future_hours[inner_index]))

            predictions.append(current_prediction.iloc[0])

            truth.append(current)

            encountered_data.append(current)
            encountered_hours.append(future_hours[inner_index])

            if len(encountered_data) > 1500:
                del encountered_data[0]
                del encountered_hours[0]

            # if index % 2 == 0:
            forecaster.fit(y=pd.Series(encountered_data), exog=pd.Series(encountered_hours))

        mse = (np.square(np.asarray(predictions) - np.asarray(truth))).mean()
        non_backtest_metric.append(np.sqrt(mse))

        print("Custom mse:", mse)

        metric, predictions_backtest = backtesting_forecaster(
            forecaster=forecaster,
            y=data,
            initial_train_size=len(data_train_test),
            fixed_train_size=False,
            steps=24,
            metric="mean_squared_error",
            refit=False,
            verbose=False
        )

        metric_list.append(np.sqrt(metric))


        print(f'Backtest error: {np.sqrt(metric)}')


        fig, ax = plt.subplots(figsize=(12, 3.5))
        pd.Series(truth).plot(ax=ax, linewidth=2, label='real')
        pd.Series(np.asarray(predictions)).plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title(f'Prediction vs real non shiftable load of house {index+1}')
        ax.legend()

        plt.show()

    print(metric_list)
    print(non_backtest_metric)


def auto_regressor_test_generic_start_exo_solar():
    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\solar_data.csv")

    load_df[load_df.columns] = MinMaxScaler().fit_transform(load_df[load_df.columns])

    data_split = np.split(load_df, 5)

    load_df_data = load_df.drop(["solar_generation_future"], axis=1)
    load_df_target = load_df["solar_generation_future"]
    hours_df = load_df["hour"]
    hours_df_split = np.split(hours_df, 5)

    direct_solar_irradiance_df = np.split(load_df["direct_solar_irradiance"], 5)
    diffuse_solar_irradiance_df = np.split(load_df["diffuse_solar_irradiance"], 5)

    total_records = 8758

    test_fraction = 0.2
    test_count = round(8758 * test_fraction)

    validation_fraction = 0.7
    validation_count = round(8758 * validation_fraction)

    metric_list = []
    non_backtest_metric = []

    for index, data in enumerate(np.split(load_df_target, 5)):

        print(f"------House {index + 1}-----")

        data_train_val = data.iloc[:validation_count]
        data_train_test = data.iloc[:test_count]

        lag_size = 60

        data_minimal = data.iloc[:lag_size]
        hours_minimal = hours_df_split[index].iloc[:lag_size]

        direct_minimal = direct_solar_irradiance_df[index].iloc[:lag_size]
        diffuse_minimal = diffuse_solar_irradiance_df[index].iloc[:lag_size]

        encountered_data = list(data_minimal)
        encountered_hours = list(hours_minimal)

        encountered_direct = list(direct_minimal)
        encountered_diffuse = list(diffuse_minimal)

        future_hours = list(hours_df_split[index].iloc[lag_size:])

        future_direct = list(direct_solar_irradiance_df[index].iloc[lag_size:])
        future_diffuse = list(diffuse_solar_irradiance_df[index].iloc[lag_size:])

        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=[1,2,3,4,5,23,24,25,26,27, 48, 49, 50, 51, 52],
            transformer_y=StandardScaler()
        )

        #forecaster.fit(y=pd.Series(data_minimal.values), exog=pd.Series(encountered_hours))
        forecaster.fit(y=pd.Series(data_minimal.values), exog=pd.Series(encountered_direct))
        #forecaster.fit(y=pd.Series(data_minimal.values), exog=pd.Series(encountered_diffuse))

        predictions = []
        truth = []

        for inner_index, current in enumerate(data.iloc[lag_size:]):

            if inner_index % 200 == 0:
                print(inner_index)

            #current_prediction = forecaster.predict(steps=1, exog=pd.Series(future_hours[inner_index]))
            current_prediction = forecaster.predict(steps=1, exog=pd.Series(future_direct[inner_index]))
            #current_prediction = forecaster.predict(steps=1, exog=pd.Series(future_diffuse[inner_index]))

            predictions.append(np.clip(current_prediction.iloc[0], 0, None))

            truth.append(current)

            encountered_data.append(current)
            #encountered_hours.append(future_hours[inner_index])
            encountered_direct.append(future_direct[inner_index])
            #encountered_diffuse.append(future_diffuse[inner_index])

            if len(encountered_data) > 1500:
                del encountered_data[0]
                #del encountered_hours[0]
                del encountered_direct[0]
                #del encountered_diffuse[0]

            # if index % 2 == 0:
            #forecaster.fit(y=pd.Series(encountered_data), exog=pd.Series(encountered_hours))
            forecaster.fit(y=pd.Series(encountered_data), exog=pd.Series(encountered_direct))
            #forecaster.fit(y=pd.Series(encountered_data), exog=pd.Series(encountered_diffuse))

        mse = (np.square(np.asarray(predictions) - np.asarray(truth))).mean()
        non_backtest_metric.append(np.sqrt(mse))

        print("Custom mse:", mse)

        metric, predictions_backtest = backtesting_forecaster(
            forecaster=forecaster,
            y=data,
            initial_train_size=len(data_train_test),
            fixed_train_size=False,
            steps=24,
            metric="mean_squared_error",
            refit=False,
            verbose=False
        )

        metric_list.append(np.sqrt(metric))


        print(f'Backtest error: {np.sqrt(metric)}')


        fig, ax = plt.subplots(figsize=(12, 3.5))
        pd.Series(truth).plot(ax=ax, linewidth=2, label='real')
        pd.Series(np.asarray(predictions)).plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title(f'Prediction vs real non shiftable load of house {index+1}')
        ax.legend()

        plt.show()

    print(metric_list)
    print(non_backtest_metric)


def auto_regressor_test_generic_start():
    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv")

    load_df[load_df.columns] = MinMaxScaler().fit_transform(load_df[load_df.columns])
    #load_df[load_df.columns] = np.log1p(load_df[load_df.columns])

    load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
    load_df_target = load_df["non_shiftable_load_future"]

    total_records = 8758

    test_fraction = 0.2
    test_count = round(8758 * test_fraction)

    validation_fraction = 0.7
    validation_count = round(8758 * validation_fraction)

    metric_list = []
    non_backtest_metric = []

    for index, data in enumerate(np.split(load_df_target, 5)):

        print(f"------House {index + 1}-----")

        data_train_val = data.iloc[:validation_count]
        data_train_test = data.iloc[:test_count]

        lag_size = 60

        data_minimal = data.iloc[:lag_size]

        encountered_data = list(data_minimal)

        transformer_y = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)

        forecaster = ForecasterAutoreg(
            regressor=Ridge(),
            lags=[1,2,3,4,5,23,24,25,26,27, 48, 49, 50, 51, 50],
            transformer_y=transformer_y
        )

        forecaster.fit(pd.Series(data_minimal.values))

        predictions = []
        truth = []

        for inner_index, current in enumerate(data.iloc[lag_size:]):

            if inner_index % 200 == 0:
                print(inner_index)

            current_prediction = forecaster.predict(steps=1, last_window=pd.Series(encountered_data[-51:]))

            #predictions.append(np.clip(current_prediction.iloc[0], 0, None))
            predictions.append(current_prediction)

            truth.append(np.clip(current, 0, None))

            encountered_data.append(current)

            if len(encountered_data) > 1500:
                del encountered_data[0]

            # if index % 2 == 0:

            if inner_index % 5 == 0:
                forecaster.fit(y=pd.Series(encountered_data))

        mse = (np.square(np.asarray(predictions) - np.asarray(truth))).mean()
        non_backtest_metric.append(np.sqrt(mse))

        print("Custom mse:", mse)

        metric, predictions_backtest = backtesting_forecaster(
            forecaster=forecaster,
            y=data,
            initial_train_size=len(data_train_test),
            fixed_train_size=False,
            steps=24,
            metric="mean_squared_error",
            refit=False,
            verbose=False
        )

        metric_list.append(np.sqrt(metric))


        print(f'Backtest error: {np.sqrt(metric)}')


        fig, ax = plt.subplots(figsize=(12, 3.5))
        pd.Series(truth).plot(ax=ax, linewidth=2, label='real')
        pd.Series(np.asarray(predictions)).plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title(f'Prediction vs real non shiftable load of house {index+1}')
        ax.legend()

        plt.show()

        # plt.plot(np.asarray(predictions)[1000:1100] - np.asarray(truth)[1000:1100])
        #
        # plt.show()

    print(metric_list)
    print(non_backtest_metric)


def auto_regressor_test_generic_start_solar():
    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\solar_data.csv")

    load_df[load_df.columns] = MinMaxScaler().fit_transform(load_df[load_df.columns])

    load_df_data = load_df.drop(["solar_generation_future"], axis=1)
    load_df_target = load_df["solar_generation_future"]

    total_records = 8758

    test_fraction = 0.2
    test_count = round(8758 * test_fraction)

    validation_fraction = 0.7
    validation_count = round(8758 * validation_fraction)

    metric_list = []
    non_backtest_metric = []

    for index, data in enumerate(np.split(load_df_target, 5)):

        print(f"------House {index + 1}-----")

        data_train_val = data.iloc[:validation_count]
        data_train_test = data.iloc[:test_count]

        lag_size = 60

        data_minimal = data.iloc[:lag_size]

        encountered_data = list(data_minimal)

        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=[1,2,3,4,5,23,24,25,26,27, 48, 49, 50, 51, 50],
            transformer_y=StandardScaler()
        )

        forecaster.fit(pd.Series(data_minimal.values))

        predictions = []
        truth = []

        for inner_index, current in enumerate(data.iloc[lag_size:]):

            if inner_index % 200 == 0:
                print(inner_index)

            current_prediction = forecaster.predict(steps=1, last_window=pd.Series(encountered_data[-51:]))

            predictions.append(current_prediction.iloc[0])

            truth.append(current)

            encountered_data.append(current)

            if len(encountered_data) > 1500:
                del encountered_data[0]

            # if index % 2 == 0:

            if inner_index % 5 == 0:
                forecaster.fit(y=pd.Series(encountered_data))

        mse = (np.square(np.asarray(predictions) - np.asarray(truth))).mean()
        non_backtest_metric.append(np.sqrt(mse))

        print("Custom mse:", mse)

        metric, predictions_backtest = backtesting_forecaster(
            forecaster=forecaster,
            y=data,
            initial_train_size=len(data_train_test),
            fixed_train_size=False,
            steps=24,
            metric="mean_squared_error",
            refit=False,
            verbose=False
        )

        metric_list.append(np.sqrt(metric))


        print(f'Backtest error: {np.sqrt(metric)}')


        fig, ax = plt.subplots(figsize=(12, 3.5))
        pd.Series(truth).plot(ax=ax, linewidth=2, label='real')
        pd.Series(np.asarray(predictions)).plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title(f'Prediction vs real solar generation of house {index+1}')
        ax.legend()

        plt.show()

    print(metric_list)
    print(non_backtest_metric)


def auto_regressor_test_custom_predictor():
    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv")

    load_df[load_df.columns] = MinMaxScaler().fit_transform(load_df[load_df.columns])

    load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
    load_df_target = load_df["non_shiftable_load_future"]

    count = 0

    while True:

        fig, ax = plt.subplots(figsize=(7, 3))

        plot_pacf(load_df_target[count:], ax=ax, lags=49)

        plt.savefig(rf"C:\Users\Lars\Documents\Epoch\CityLearn\CityLearn Plots\Partial_Auto\img_{count}_.png")

        plt.clf()

        count += 1

    sys.exit()

    total_records = 8758

    test_fraction = 0.2
    test_count = round(8758 * test_fraction)

    validation_fraction = 0.7
    validation_count = round(8758 * validation_fraction)

    metric_list = []
    non_backtest_metric = []

    for index, data in enumerate(np.split(load_df_target, 5)):

        print(f"------House {index + 1}-----")

        data_train_val = data.iloc[:validation_count]
        data_train_test = data.iloc[:test_count]

        lag_size = 60

        data_minimal = data.iloc[:lag_size]

        encountered_data = list(data_minimal)

        # forecaster = ForecasterAutoreg(
        #     regressor=Ridge(random_state=42),
        #     lags=[1,2,3,4,5,23,24,25,26,27, 48, 49, 50, 51, 50],
        #     transformer_y=StandardScaler()
        # )

        def custom_predictor(y):
            lags = y[-1:-50:-1]
            mean = np.mean(y[-10:])
            predictors = np.hstack([lags, mean])

            return predictors

        forecaster = ForecasterAutoregCustom(
            regressor=Ridge(random_state=42),
            fun_predictors=custom_predictor,
            window_size=50,
        )

        forecaster.fit(pd.Series(data_minimal.values))

        predictions = []
        truth = []

        for inner_index, current in enumerate(data.iloc[lag_size:]):

            if inner_index % 200 == 0:
                print(inner_index)

            current_prediction = forecaster.predict(steps=1, last_window=pd.Series(encountered_data[-51:]))

            predictions.append(current_prediction.iloc[0])

            truth.append(current)

            encountered_data.append(current)

            if len(encountered_data) > 1500:
                del encountered_data[0]

            # if index % 2 == 0:

            if inner_index % 5 == 0:
                forecaster.fit(y=pd.Series(encountered_data))

        mse = (np.square(np.asarray(predictions) - np.asarray(truth))).mean()
        non_backtest_metric.append(np.sqrt(mse))

        print("Custom mse:", mse)

        metric, predictions_backtest = backtesting_forecaster(
            forecaster=forecaster,
            y=data,
            initial_train_size=len(data_train_test),
            fixed_train_size=False,
            steps=24,
            metric="mean_squared_error",
            refit=False,
            verbose=False
        )

        metric_list.append(np.sqrt(metric))


        print(f'Backtest error: {np.sqrt(metric)}')


        fig, ax = plt.subplots(figsize=(12, 3.5))
        pd.Series(truth).plot(ax=ax, linewidth=2, label='real')
        pd.Series(np.asarray(predictions)).plot(linewidth=2, label='prediction', ax=ax)
        ax.set_title(f'Prediction vs real non shiftable load of house {index+1}')
        ax.legend()

        plt.show()

    print(metric_list)
    print(non_backtest_metric)


def lag_list_maker():

    days = 3
    day_slice = 20

    list_container = []

    for i in range(0, days+1):

        for j in range(1, day_slice+1):

            ls = []

            for k in range(0, i+1):

                l = list(np.arange(1 + (k * 22), j + (k * 22)))
                ls.append(l)

            flat_list = [item for sublist in ls for item in sublist]
            list_container.append(flat_list)

    print(list_container)

def correlation_finder():

    building_1_data = osp.join(osp.dirname(competition_data.__file__), "building_1.csv")
    building_2_data = osp.join(osp.dirname(competition_data.__file__), "building_2.csv")
    building_3_data = osp.join(osp.dirname(competition_data.__file__), "building_3.csv")
    building_4_data = osp.join(osp.dirname(competition_data.__file__), "building_4.csv")
    building_5_data = osp.join(osp.dirname(competition_data.__file__), "building_5.csv")

    pricing = osp.join(osp.dirname(competition_data.__file__), "pricing.csv")

    take_first = 8760

    b1_df = pd.read_csv(building_1_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b2_df = pd.read_csv(building_2_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b3_df = pd.read_csv(building_3_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b4_df = pd.read_csv(building_4_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]
    b5_df = pd.read_csv(building_5_data)["Equipment Electric Power [kWh]"].to_numpy()[0:take_first]

    price_df = pd.read_csv(pricing)["Electricity Pricing [$]"].to_numpy()[0:take_first]

    price_df = np.split(price_df, 365)

    price_df = np.asarray(price_df).mean(axis=0)

    plt.plot(np.arange(1, 25), price_df)

    plt.show()


def stan_init(m):
    """Retrieve parameters from a trained model.

    Retrieve parameters from a trained model in the format
    used to initialize a new Stan model.

    Parameters
    ----------
    m: A trained model of the Prophet class.

    Returns
    -------
    A Dictionary containing retrieved parameters of m.

    """
    res = {}
    for pname in ['k', 'm', 'sigma_obs']:
        res[pname] = m.params[pname][0][0]
    for pname in ['delta', 'beta']:
        res[pname] = m.params[pname][0]
    return res

def prophet_test():
    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv")

    load_df[load_df.columns] = MinMaxScaler().fit_transform(load_df[load_df.columns])
    #load_df[load_df.columns] = np.log1p(load_df[load_df.columns])

    load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
    load_df_target = load_df["non_shiftable_load_future"]

    total_records = 8758

    mse_list = []
    mae_list = []

    import logging
    logging.getLogger('prophet').setLevel(logging.ERROR)

    for index, data in enumerate(np.split(load_df_target, 5)):

        print(f"------House {index + 1}-----")

        base = pd.Timestamp.today()

        timestamp_list = [base + datetime.timedelta(hours=x) for x in range(len(data))]

        pr_df = pd.DataFrame({"ds": timestamp_list, "y": data})

        lag_size = 60

        data_minimal = pr_df.head(lag_size)

        m = Prophet(growth="flat")

        m.fit(data_minimal)

        encountered_index = 0

        predictions = []
        truths = []

        while encountered_index + lag_size < total_records:

            future = m.make_future_dataframe(periods=1)

            forecast = m.predict(future)

            prediction = forecast[["yhat"]].tail(1).iloc[0,0]

            truth = pr_df.iloc[lag_size + encountered_index]["y"]

            predictions.append(prediction)

            truths.append(truth)

            encountered_index += 1

            #print(pr_df.head(encountered_index + lag_size))

            m = Prophet(growth="flat").fit(pr_df.head(encountered_index + lag_size))

            if encountered_index % 10 == 0:
                print(encountered_index)

                mae = metrics.mean_absolute_error(truths, predictions)
                mse = metrics.mean_squared_error(truths, predictions)

                print("mae:", mae)
                print("mse", mse)

        # mae_list.append(mae)
        # mse_list.append(mse)

    print(mae_list)
    print(mse_list)



if __name__ == '__main__':
    # plot_average_lf()
    # day_difference()
    # average_difference()
    # auto_regressor_test_grid()
    # auto_regressor_test_non_grid()

    # lag_list_maker()
    #auto_regressor_test_generic_start()
    #auto_regressor_test_generic_start_exo_solar()
    auto_regressor_test_generic_start_solar()
    #auto_regressor_test_custom_predictor()
    #correlation_finder()

    #prophet_test()
