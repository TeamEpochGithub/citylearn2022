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


def switched_data_avg(hours: int, buildings: list, data):
    b = pd.concat(buildings)
    b_avg = b.groupby(b.index).mean()
    b_avg = b_avg.shift(-hours)
    b_avg = b_avg.dropna()

    data.drop(data.index[[*range(-1, -hours - 1, -1)]], inplace=True, axis=0)

    return b_avg, data


def find_coefficients(buildings: list):
    b = pd.concat(buildings)
    b_avg = b.groupby(b.index).mean()

    # b_avg = b_avg.loc[(b_avg != 0).any(axis=1)] #Doesn't work
    b_avg = b_avg.replace({0: np.NaN})
    b_avg = b_avg.dropna()

    coefficients = []

    for building in buildings:
        # b_no0 = building.loc[(building != 0).any(axis=1)]
        b_no0 = building.replace({0: np.NaN})
        b_no0 = b_no0.dropna()

        b_diff = (b_no0 / b_avg)
        coefficients.append(b_diff.mean())

    return coefficients


def coeff_accuracy(coefficients: list, buildings: list, average):
    accuracy = []
    if type(average) == list:
        avg = np.array(average)
    else:
        avg = average.to_numpy()

    for i in range(len(coefficients)):
        building = np.array(buildings[i])
        predictions = avg * coefficients[i]
        errors = np.divide(predictions - building, building, out=np.zeros_like(predictions), where=building != 0)
        errors = np.absolute(errors)
        accuracy.append(1 - np.average(errors))

    return accuracy


b = pd.concat(builds)
b_avg = b.groupby(b.index).mean()
print(coeff_accuracy(find_coefficients(builds), builds, b_avg))
