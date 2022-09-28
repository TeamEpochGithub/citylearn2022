import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self):
        self.prices = self.load_price_df()
        self.carbon = self.load_carbon_df()

        self.mean_prices = np.mean(self.prices)
        self.mean_carbon = np.mean(self.carbon)

    @staticmethod
    def load_price_df():
        return pd.read_csv("../data/citylearn_challenge_2022_phase_1/pricing.csv")[
            'Electricity Pricing [$]']

    @staticmethod
    def load_carbon_df():
        return pd.read_csv("../data/citylearn_challenge_2022_phase_1/carbon_intensity.csv")[
            'kg_CO2/kWh']

    def get_mean_prices(self):
        return self.mean_prices

    def get_price(self, hour: int):
        return self.prices[hour]

    def get_carbon(self, hour: int):
        return self.carbon[hour]

    def get_mean_carbon(self):
        return self.mean_carbon

    def get_cost_list(self):
        return [x + y for x, y in zip(self.prices, self.carbon)]

    # @staticmethod
    # def get_usage_list(building):
    #     return \
    #         pd.read_csv(
    #             f"../data/citylearn_challenge_2022_phase_1/consumptions.csv")[
    #             f"B{building + 1}"]
