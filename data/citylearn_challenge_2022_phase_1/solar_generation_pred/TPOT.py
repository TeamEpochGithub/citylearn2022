from tpot import TPOTRegressor
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_boston

if __name__ == "__main__":

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
    training, testing, training_labels, testing_labels = train_test_split(x, y, test_size=.25, random_state=42)

    opt = TPOTRegressor(verbosity=2, random_state=42, scoring="r2")
    opt.fit(training, training_labels)
    print(opt.score(testing, testing_labels))
    opt.export("tpot_pipeline.py")