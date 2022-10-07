import numpy as np
import pandas as pd
import os.path as osp
from data import citylearn_challenge_2022_phase_1 as competition_data
import building_action_logs as logs

class decision_object():

    def __init__(self, hour, action, score):

        self.hour = hour
        self.action = action
        self.score = score


def search(hour, depth):

    assert 0 <= hour < 24
    assert 0 < depth < 25


def setup():

    weather_data = osp.join(osp.dirname(competition_data.__file__), "weather.csv")
    carbon_data = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")
    pricing_data = osp.join(osp.dirname(competition_data.__file__), "pricing.csv")

    building_1_data = osp.join(osp.dirname(competition_data.__file__), "building_1.csv")
    building_2_data = osp.join(osp.dirname(competition_data.__file__), "building_2.csv")
    building_3_data = osp.join(osp.dirname(competition_data.__file__), "building_3.csv")
    building_4_data = osp.join(osp.dirname(competition_data.__file__), "building_4.csv")
    building_5_data = osp.join(osp.dirname(competition_data.__file__), "building_5.csv")

    b_1_df = pd.read_csv(building_1_data)

    custom_b1_df = pd.DataFrame(columns=["hour", "best action", "split size", "soc", "capacity"])
    custom_b1_df["hour"] = np.arange(1, 8760)

    print(custom_b1_df)


if __name__ == '__main__':
    setup()
