import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from joblib import dump, load



if __name__ == '__main__':
    solar_df = pd.read_csv(
        r"C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\consumption_prediction\solar_data.csv")

    solar_df_data = solar_df.drop(["solar_generation_future"], axis=1)
    solar_df_target = solar_df["solar_generation_future"]

    X_train, X_test, y_train, y_test = train_test_split(solar_df_data.to_numpy(), solar_df_target.to_numpy(),
                                                        train_size=0.8, test_size=0.2)

    # Average CV score on the training set was: -0.020452843615494536
    exported_pipeline = make_pipeline(
        StackingEstimator(
            estimator=SGDRegressor(alpha=0.0, eta0=0.1, fit_intercept=True, l1_ratio=0.0, learning_rate="invscaling",
                                   loss="epsilon_insensitive", penalty="elasticnet", power_t=100.0)),
        StandardScaler(),
        GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=10,
                                  max_features=0.6500000000000001, min_samples_leaf=12, min_samples_split=15,
                                  n_estimators=100, subsample=0.8500000000000001)
    )
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 42)

    exported_pipeline.fit(X_train, y_train)

    dump(exported_pipeline, 'solar_model.joblib')

    results = exported_pipeline.predict(X_test)

    mse = (np.square(results - y_test)).mean()

    print(exported_pipeline.score(X_test, y_test))
    print(mse)
    print(results)