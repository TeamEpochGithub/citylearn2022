import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
load_df = pd.read_csv(
        r"C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\load_data.csv")

# Scaler training data
ms = MinMaxScaler()
load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
load_df_data[load_df_data.columns] = ms.fit_transform(load_df_data[load_df_data.columns])

print(load_df["non_shiftable_load_future"])
ms_result = MinMaxScaler()
load_df[["non_shiftable_load_future"]] = ms_result.fit_transform(load_df[["non_shiftable_load_future"]])
load_df_target = load_df["non_shiftable_load_future"]


print(load_df_target)

X_train, X_test, y_train, y_test = train_test_split(load_df_data.to_numpy(), load_df_target.to_numpy(),
                                                    train_size=0.8, test_size=0.2)

# Average CV score on the training set was: -0.005904836355109909
exported_pipeline = ExtraTreesRegressor(bootstrap=False, max_features=0.6000000000000001, min_samples_leaf=13, min_samples_split=6, n_estimators=100)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

dump(exported_pipeline, 'load_model.joblib')
mse = (np.square(results - y_test)).mean()

print(X_test, X_train, y_train, y_test)

print(exported_pipeline.score(X_test, y_test))
print(mse)
