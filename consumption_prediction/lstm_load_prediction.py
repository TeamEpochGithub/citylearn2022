import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import os.path as osp
from data import citylearn_challenge_2022_phase_1 as competition_data

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
plt.style.use('fivethirtyeight')


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


def auto_regressor_test_generic_start():
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

        lag_size = 60

        data_minimal = data.iloc[:lag_size]

        encountered_data = list(data_minimal)

        forecaster = ForecasterAutoreg(
            regressor=Ridge(random_state=42),
            lags=[1,2,3,4,5,23,24,25,26,27, 48, 49, 50, 51, 52],
            transformer_y=StandardScaler()
        )

        forecaster.fit(pd.Series(data_minimal.values))

        predictions = []
        truth = []

        for inner_index, current in enumerate(data.iloc[lag_size:]):

            if inner_index % 200 == 0:
                print(inner_index)

            current_prediction = forecaster.predict(steps=1)

            predictions.append(current_prediction.iloc[0])

            truth.append(current)

            encountered_data.append(current)

            # if len(encountered_data) > 1500:
            #     del encountered_data[0]

            # if index % 2 == 0:
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
    day_slice = 10

    list_container = []

    for i in range(0, days+1):

        for j in range(1, day_slice+1):

            ls = []

            for k in range(0, i+1):

                l = list(np.arange(1 + (k * 24), j + (k * 24)))
                ls.append(l)

            flat_list = [item for sublist in ls for item in sublist]
            list_container.append(flat_list)

    print(list_container)


if __name__ == '__main__':
    # plot_average_lf()
    # day_difference()
    # average_difference()
    # auto_regressor_test_grid()
    # auto_regressor_test_non_grid()

    lag_list_maker()
    #auto_regressor_test_generic_start()
