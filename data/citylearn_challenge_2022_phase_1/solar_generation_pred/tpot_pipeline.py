import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from TPOT import switched_data

wthr = pd.read_csv("weather.csv")[
    ["Outdoor Drybulb Temperature [C]", "Relative Humidity [%]", "Diffuse Solar Radiation [W/m2]",
     "Direct Solar Radiation [W/m2]"]]

data = pd.concat([wthr, pd.read_csv("Building_1.csv")[["Month", "Hour"]]], axis=1)

(b1, b2, b3, b4, b5) = (pd.read_csv(f"Building_{i}.csv")["Solar Generation [W/kW]"] for i in range(1, 6))
builds = [b1, b2, b3, b4, b5]

y, x = switched_data(5, builds, data)

training, testing, training_labels, testing_labels = train_test_split(x, y, test_size=.25, random_state=42)



# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.8934149476860018
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=LinearSVR(C=1.0, dual=True, epsilon=0.001, loss="epsilon_insensitive", tol=0.0001)),
        make_union(
            Nystroem(gamma=0.4, kernel="rbf", n_components=6),
            StackingEstimator(estimator=make_pipeline(
                StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.1, loss="linear", n_estimators=100)),
                StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.5, loss="ls", max_depth=8, max_features=0.6500000000000001, min_samples_leaf=4, min_samples_split=5, n_estimators=100, subsample=1.0)),
                GradientBoostingRegressor(alpha=0.75, learning_rate=0.01, loss="huber", max_depth=7, max_features=0.3, min_samples_leaf=10, min_samples_split=17, n_estimators=100, subsample=0.4)
            ))
        )
    ),
    ExtraTreesRegressor(bootstrap=True, max_features=0.4, min_samples_leaf=13, min_samples_split=19, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training, training_labels)
results = exported_pipeline.predict(testing_features)
