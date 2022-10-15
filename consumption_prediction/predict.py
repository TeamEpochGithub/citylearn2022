import sys

import numpy as np
import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

def predict():

    pipeline_optimizer = TPOTRegressor()

    # solar_df = pd.read_csv(r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\consumption_prediction\solar_data.csv")
    #
    # solar_df_data = solar_df.drop(["solar_generation_future"], axis=1)
    # solar_df_target = solar_df["solar_generation_future"]

    load_df = pd.read_csv(
        r"C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv")

    # Scaler training data
    ms = MinMaxScaler()
    load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
    load_df_data[load_df_data.columns] = ms.fit_transform(load_df_data[load_df_data.columns])

    ms_result = MinMaxScaler()
    load_df[["non_shiftable_load_future"]] = ms_result.fit_transform(load_df[["non_shiftable_load_future"]])
    load_df_target = load_df["non_shiftable_load_future"]

    X_train, X_test, y_train, y_test = train_test_split(load_df_data.to_numpy(), load_df_target.to_numpy(),
                                                        train_size=0.8, test_size=0.2)

    print(X_train, type(X_train))
    sys.exit()

    pipeline_optimizer = TPOTRegressor(generations=20, population_size=20, cv=5,
                                        random_state=42, verbosity=2, n_jobs=-1)

    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))
    pipeline_optimizer.export('./tpot_exported_pipeline_load_small.py')

    dump(ms, 'ms_load_data')
    dump(ms_result, 'ms_load_result')


if __name__ == '__main__':
    predict()
