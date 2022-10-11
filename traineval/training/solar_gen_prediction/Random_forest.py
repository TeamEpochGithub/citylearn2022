import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

wthr = pd.read_csv("weather.csv")[
    ["Outdoor Drybulb Temperature [C]", "Relative Humidity [%]", "Diffuse Solar Radiation [W/m2]",
     "Direct Solar Radiation [W/m2]"]]
data = pd.concat([wthr, pd.read_csv("Building_1.csv")[["Month", "Hour"]]], axis=1)

(b1, b2, b3, b4, b5) = (pd.read_csv(f"Building_{i}.csv")["Solar Generation [W/kW]"] for i in range(1, 6))
builds = [b1, b2, b3, b4, b5]


def switched_data(hours: int, buildings: list, data):
    switched = []

    for b in buildings:
        switched.append(b.shift(-hours).dropna())

    generation = pd.concat(switched, ignore_index=True)

    data.drop(data.index[[*range(-1, -hours - 1, -1)]], inplace=True, axis=0)
    dataframe = pd.concat([data for i in range(len(buildings))], ignore_index=True)

    return generation, dataframe


y, x = switched_data(5, builds, data)

for feature in list(x.columns):
    training, testing, training_labels, testing_labels = train_test_split(x.drop(labels=feature, axis=1), y,
                                                                          test_size=.25, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(training, training_labels)
    preds = rf.predict(testing)
    print("Dataset" + feature + "\n" + str(rf.score(training, training_labels)) + "\n")
    print(rf.score(testing, testing_labels))

training, testing, training_labels, testing_labels = train_test_split(x, y, test_size=.25, random_state=42)

# sc = StandardScaler()
# normed_train_data = pd.DataFrame(sc.fit_transform(training), columns=x.columns)
# normed_test_data = pd.DataFrame(sc.fit_transform(testing), columns=x.columns)

rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(training, training_labels)
preds = rf.predict(testing)
# print(testing)
# print(preds)
print(rf.score(training, training_labels))
print(rf.score(testing, testing_labels))

"""feature_list = list(x.columns)
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];"""
