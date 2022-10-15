import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from tpot.export_utils import set_param_recursive
from joblib import dump

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
load_df = pd.read_csv(
        r"C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv")

# Scaler training data
ms_load = MinMaxScaler()
load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
load_df_data[load_df_data.columns] = ms_load.fit_transform(load_df_data[load_df_data.columns])

ms_load_result = MinMaxScaler()
load_df[["non_shiftable_load_future"]] = ms_load_result.fit_transform(load_df[["non_shiftable_load_future"]])
load_df_target = load_df["non_shiftable_load_future"]

X_train, X_test, y_train, y_test = train_test_split(load_df_data.to_numpy(), load_df_target.to_numpy(),
                                                    train_size=0.8, test_size=0.2)

# Average CV score on the training set was: -0.005864248219567833
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    ExtraTreesRegressor(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=3, min_samples_split=18, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

dump(exported_pipeline, '../prediction_models/load_model.joblib')
dump(ms_load, '../prediction_models/ms_load_data.joblib')
dump(ms_load_result, '../prediction_models/ms_load_result.joblib')
mse = (np.square(results - y_test)).mean()

print(X_test, X_train, y_train, y_test)

print(exported_pipeline.score(X_test, y_test))
print(mse)
