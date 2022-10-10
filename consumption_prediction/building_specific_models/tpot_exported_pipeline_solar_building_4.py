import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
solar_df = pd.read_csv(
        r"C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\consumption_prediction\building_specific_models\building4_solar.csv")

solar_df_data = solar_df.drop(["solar_generation_future"], axis=1)
solar_df_target = solar_df["solar_generation_future"]

X_train, X_test, y_train, y_test = train_test_split(solar_df_data.to_numpy(), solar_df_target.to_numpy(),
                                                    train_size=0.8, test_size=0.2)
# Average CV score on the training set was: -0.028300897591498852
exported_pipeline = ExtraTreesRegressor(bootstrap=True, max_features=0.8500000000000001, min_samples_leaf=2, min_samples_split=2, n_estimators=100)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(X_train, y_train)
results = exported_pipeline.predict(X_test)

mse = (np.square(results - y_test)).mean()

print(exported_pipeline.score(X_test, y_test))
print(mse)
