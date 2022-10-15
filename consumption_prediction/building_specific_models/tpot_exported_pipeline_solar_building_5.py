import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
solar_df = pd.read_csv(
        r"C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\consumption_prediction\building_specific_models\building5_solar.csv")

solar_df_data = solar_df.drop(["solar_generation_future"], axis=1)
solar_df_target = solar_df["solar_generation_future"]

X_train, X_test, y_train, y_test = train_test_split(solar_df_data.to_numpy(), solar_df_target.to_numpy(),
                                                    train_size=0.8, test_size=0.2)

# Average CV score on the training set was: -0.03076016731505261
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=0.1, fit_intercept=False, l1_ratio=0.5, learning_rate="constant", loss="huber", penalty="elasticnet", power_t=0.5)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.8, min_samples_leaf=2, min_samples_split=11, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

mse = (np.square(results - y_test)).mean()

print(exported_pipeline.score(X_test, y_test))
print(mse)