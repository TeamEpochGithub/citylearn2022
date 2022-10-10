import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    load_df = pd.read_csv(
            r"C:\Users\Lars\Documents\Epoch\CityLearn\citylearn-2022-starter-kit\consumption_prediction\load_data.csv")

    load_df_data = load_df.drop(["non_shiftable_load_future"], axis=1)
    load_df_target = load_df["non_shiftable_load_future"]

    X_train, X_test, y_train, y_test = train_test_split(load_df_data.to_numpy(), load_df_target.to_numpy(),
                                                        train_size=0.8, test_size=0.2)

    # Average CV score on the training set was: -0.3756606023115455
    exported_pipeline = RandomForestRegressor(bootstrap=True, max_features=0.4, min_samples_leaf=16, min_samples_split=14, n_estimators=100)
    # Fix random state in exported estimator
    if hasattr(exported_pipeline, 'random_state'):
        setattr(exported_pipeline, 'random_state', 42)

    exported_pipeline.fit(X_train, y_train)
    results = exported_pipeline.predict(X_test)

    mse = (np.square(results - y_test)).mean()

    print(exported_pipeline.score(X_test, y_test))
    print(mse)
