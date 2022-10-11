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
y = tpot_data.iloc[:, -1]
X = tpot_data.iloc[:, :-1]

training_features, testing_features, training_target, testing_target = \
    train_test_split(X, y, train_size=0.99, random_state=None)

# Average CV score on the training set was: -0.0002613516137840111
exported_pipeline = make_pipeline(
    StackingEstimator(
        estimator=XGBRegressor(learning_rate=1.0, max_depth=4, min_child_weight=7, n_estimators=100, n_jobs=1,
                               objective="reg:squarederror", subsample=0.5, verbosity=0)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    KNeighborsRegressor(n_neighbors=4, p=1, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
print(exported_pipeline.score(testing_features, testing_target))
import joblib
joblib.dump(exported_pipeline, 'pipe.joblib')