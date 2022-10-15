import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
import os.path as osp
import data.citylearn_challenge_2022_phase_1 as competition_data

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
action_file_path = osp.join(osp.dirname(competition_data.__file__), 'perfect_actions.csv')

tpot_data = pd.read_csv(action_file_path, sep=',', dtype=np.float64)
X = tpot_data[["month",
    "day_type",
    "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_relative_humidity",
    "diffuse_solar_irradiance",
    "direct_solar_irradiance",
    "carbon_intensity",
    "non_shiftable_load",
    "solar_generation",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing"]]
y = tpot_data[["action"]]

training_features, testing_features, training_target, testing_target = \
    train_test_split(X, y, train_size=0.999, random_state=None)

# Average CV score on the training set was: -0.0032779050501836257
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    RandomForestRegressor(bootstrap=True, max_features=0.5, min_samples_leaf=18, min_samples_split=12, n_estimators=100)
)

exported_pipeline.fit(training_features.values, training_target.values)
results = exported_pipeline.predict(testing_features.values)

import joblib
joblib.dump(exported_pipeline, 'pipe.joblib')

