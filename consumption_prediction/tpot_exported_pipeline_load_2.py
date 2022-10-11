import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MaxAbsScaler
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
if __name__ == '__main__':
    solar_df = pd.read_csv(r"C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\consumption_prediction\load_data.csv")

    solar_df_data = solar_df.drop(["non_shiftable_load_future"], axis=1)
    solar_df_target = solar_df["non_shiftable_load_future"]

    X_train, X_test, y_train, y_test = train_test_split(solar_df_data.to_numpy(), solar_df_target.to_numpy(),
                                                        train_size=0.8, test_size=0.2)

# Average CV score on the training set was: -0.37443479937647794
    exported_pipeline = make_pipeline(
        StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.55, min_samples_leaf=18, min_samples_split=6, n_estimators=100)),
        VarianceThreshold(threshold=0.05),
        MaxAbsScaler(),
        ExtraTreesRegressor(bootstrap=False, max_features=0.35000000000000003, min_samples_leaf=13, min_samples_split=12, n_estimators=100)
    )
    # Fix random state for all the steps in exported pipeline
    exported_pipeline.fit(X_train, y_train)
    results = exported_pipeline.predict(X_test)

    mse = (np.square(results - y_test)).mean()

    print(exported_pipeline.score(X_test, y_test))
    print(mse)
