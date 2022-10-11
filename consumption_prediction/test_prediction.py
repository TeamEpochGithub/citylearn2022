import joblib
import pandas as pd
import numpy as np
from joblib import load
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive


solar_df = pd.read_csv('./building_specific_models/building1_load.csv')
solar_df_data = solar_df.drop(["non_shiftable_load_future"], axis=1)
solar_df_target = solar_df["non_shiftable_load_future"]

X_train, X_test, y_train, y_test = train_test_split(solar_df_data.to_numpy(), solar_df_target.to_numpy(),
                                                        train_size=0.8, test_size=0.2, random_state=2)

model = load('./building_specific_models/building1_load.joblib')

results = model.predict(X_test)
mse = (np.square(results - y_test)).mean()

print(X_test, X_train, y_train, y_test)
print(model.score(X_test, y_test))
print(mse)
print(results)
