import sys

import numpy as np
import pandas as pd
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

def predict():

    pipeline_optimizer = TPOTRegressor()

    solar_df = pd.read_csv(r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\consumption_prediction\solar_data.csv")

    solar_df_data = solar_df.drop(["solar_generation_future"], axis=1)
    solar_df_target = solar_df["solar_generation_future"]

    load_df = pd.read_csv(
        r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\consumption_prediction\load_data.csv")

    load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
    load_df_target = load_df["non_shiftable_load_future"]

    X_train, X_test, y_train, y_test = train_test_split(load_df_data.to_numpy(), load_df_target.to_numpy(),
                                                        train_size=0.8, test_size=0.2)

    pipeline_optimizer = TPOTRegressor(generations=5, population_size=20, cv=5,
                                        random_state=42, verbosity=2)

    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))
    pipeline_optimizer.export('tpot_exported_pipeline_load_1.py')


if __name__ == '__main__':
    predict()
