import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

solar_df = pd.read_csv(r'C:\Users\kuipe\OneDrive\Bureaublad\Epoch\citylearn-2022-starter-kit\consumption_prediction\load_data.csv')

params = ['outdoor_dry_bulb', 'outdoor_relative_humidity', 'carbon_intensity', 'non_shiftable_load', 'non_shiftable_load_future']
ts_data = solar_df[params].copy()

values = ts_data.values

groups = [0, 1, 2, 3, 4]
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(ts_data.columns[group], y=0.5, loc='right')
    i += 1
plt.show()

values[:, 1]

from sklearn.model_selection import train_test_split

solar_df_data = ts_data.drop(["non_shiftable_load_future"], axis=1)
solar_df_target = ts_data["non_shiftable_load_future"]

X_train, X_test, y_train, y_test = train_test_split(solar_df_data.to_numpy(), solar_df_target.to_numpy(),
                                                    train_size=0.8, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model = Sequential()
model.add(LSTM(50))
model.add(Dense(5))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(X_train, y_train, epochs=200, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()