import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as mp

#Calculate some ranges for price emission and grid
price_df = pd.read_csv("../data/citylearn_challenge_2022_phase_1/pricing.csv")
carbon_df = pd.read_csv("../data/citylearn_challenge_2022_phase_1/carbon_intensity.csv")
weather_df = pd.read_csv("../data/citylearn_challenge_2022_phase_1/weather.csv")

#Building dfs
building_1 = pd.read_csv("../data/citylearn_challenge_2022_phase_1/Building_1.csv")
building_2 = pd.read_csv("../data/citylearn_challenge_2022_phase_1/Building_2.csv")
building_3 = pd.read_csv("../data/citylearn_challenge_2022_phase_1/Building_3.csv")
building_4 = pd.read_csv("../data/citylearn_challenge_2022_phase_1/Building_4.csv")
building_5 = pd.read_csv("../data/citylearn_challenge_2022_phase_1/Building_5.csv")



#percentile electricity pricing


if __name__ == "__main__":
    # print(price_df.quantile([.33, .66]).to_string())
    # print(carbon_df.quantile([.33, .66]).to_string())
    print(weather_df.quantile([.33, .66]).to_string())

    buildings_df = pd.concat([building_1, building_2, building_3, building_4, building_5]).groupby(level=0).mean()
    print(buildings_df.quantile([.33, .66]).to_string())

    # print(buildings_df[0:100].to_string())

