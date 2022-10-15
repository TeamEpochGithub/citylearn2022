import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
solar_df = pd.read_csv(
        r"C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\data\citylearn_challenge_2022_phase_1\solar_data.csv")

# Scaler training data
ms_solar = MinMaxScaler()
solar_df_data = solar_df.drop(["solar_generation_future"], axis=1)
solar_df_data[solar_df_data.columns] = ms_solar.fit_transform(solar_df_data[solar_df_data.columns])

ms_solar_result = MinMaxScaler()
solar_df[["solar_generation_future"]] = ms_solar_result.fit_transform(solar_df[["solar_generation_future"]])
solar_df_target = solar_df["solar_generation_future"]

X_train, X_test, y_train, y_test = train_test_split(solar_df_data.to_numpy(), solar_df_target.to_numpy(),
                                                    train_size=0.8, test_size=0.2)

# Average CV score on the training set was: -0.0012961075388285526
exported_pipeline = GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=10, max_features=0.6500000000000001, min_samples_leaf=10, min_samples_split=8, n_estimators=100, subsample=1.0)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

dump(exported_pipeline, '../prediction_models/solar_model.joblib')
dump(ms_solar, '../prediction_models/ms_solar_data.joblib')
dump(ms_solar_result, '../prediction_models/ms_solar_result.joblib')
mse = (np.square(results - y_test)).mean()

print(X_test, X_train, y_train, y_test)

print(exported_pipeline.score(X_test, y_test))
print(mse)
