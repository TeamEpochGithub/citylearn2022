import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from joblib import dump

# NOTE: Make sure that the outcome column is labeled 'target' in the analysis_data file
solar_df = pd.read_csv('./building3_load.csv')
solar_df_data = solar_df.drop(["non_shiftable_load_future"], axis=1)
solar_df_target = solar_df["non_shiftable_load_future"]

X_train, X_test, y_train, y_test = train_test_split(solar_df_data.to_numpy(), solar_df_target.to_numpy(),
                                                        train_size=0.8, test_size=0.2, random_state=2)

# Average CV score on the training set was: -0.21941074840807548
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=12, min_samples_split=8, n_estimators=100)),
    ExtraTreesRegressor(bootstrap=True, max_features=0.7000000000000001, min_samples_leaf=3, min_samples_split=8, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)


exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

dump(exported_pipeline, 'building3_load.joblib')
mse = (np.square(results - y_test)).mean()
print(mse)