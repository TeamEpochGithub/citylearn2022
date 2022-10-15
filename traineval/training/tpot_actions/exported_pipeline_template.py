import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV, RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
import os.path as osp
import data.citylearn_challenge_2022_phase_1 as competition_data
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)

# NOTE: Make sure that the outcome column is labeled 'target' in the data file

action_file_path = osp.join(osp.dirname(competition_data.__file__), 'perfect_actions.csv')

tpot_data = pd.read_csv(action_file_path, sep=',', dtype=np.float64)

col_names = ["month",
             "day_type",
             "hour",
             "non_shiftable_load",
             "solar_generation",
             "electrical_storage_soc",
             "net_electricity_consumption",
             "action"]

num_consumptions = 6
consumption_col_names = []
for i in range(num_consumptions):
    consumption_col_names.append(f"consumption_hour{i + 1}")

all_col_names = col_names + consumption_col_names

y = tpot_data[["action"]]
X = tpot_data[all_col_names].drop(["action"], axis=1)

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

groups = np.array([i % 5 for i in range(len(X))])
logo = LeaveOneGroupOut()
cv_iter = list(logo.split(X, y, groups))

training_features, testing_features, training_target, testing_target = \
    train_test_split(X, y, train_size=0.999, random_state=None)

exported_pipeline = ExtraTreesRegressor(bootstrap=True, max_features=0.9500000000000001, min_samples_leaf=3,
                                        min_samples_split=7, n_estimators=100)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

from sklearn.metrics import mean_squared_error

print(mean_squared_error(exported_pipeline.predict(testing_features), testing_target))

import joblib

joblib.dump(exported_pipeline, 'pipe.joblib')
