import numpy as np
import pandas as pd
import itertools

import os
import sys



def evaluate_data():

    #pd.set_option('display.max_columns', None)

    base_directory = os.path.dirname(os.path.realpath(__file__))

    dir = f"{base_directory}/data/citylearn_challenge_2022_phase_1/"

    weather_file = f"{dir}weather.csv"
    pricing_file = f"{dir}pricing.csv"
    carbon_intensity_file = f"{dir}carbon_intensity.csv"

    w_df = pd.read_csv(weather_file)
    w_df = w_df[["Diffuse Solar Radiation [W/m2]", "Direct Solar Radiation [W/m2]"]]
    w_df = w_df.sum(axis=1)

    p_df = pd.read_csv(pricing_file)
    p_df = p_df[["Electricity Pricing [$]"]]

    ci_df = pd.read_csv(carbon_intensity_file)


    state_size = 5
    action_space_size = 10
    float_rounding = 2


    carbon_intervals = np.ndarray.round(np.linspace(ci_df.min(), ci_df.max(), state_size), float_rounding)
    pricing_intervals = np.ndarray.round(np.linspace(p_df.min(), p_df.max(), state_size), float_rounding)

    solar_intervals = list(np.ndarray.round(np.linspace(w_df.min(), w_df.max(), state_size), 0))
    solar_intervals = [int(x) for x in solar_intervals]

    pricing_intervals = list(itertools.chain.from_iterable(pricing_intervals))
    carbon_intervals = list(itertools.chain.from_iterable(carbon_intervals))

    action_space = list(np.ndarray.round(np.linspace(-1, 1, action_space_size), 2))

    row_indeces = []

    for price in pricing_intervals:

        for carbon in carbon_intervals:

            for solar in solar_intervals:

                row_indeces.append(f"p_{price}_c_{carbon}_s_{solar}")

    q_values = pd.DataFrame(0, index=row_indeces, columns=action_space)

    q_values.to_csv(f"{base_directory}/q_tables/p_c_s_size_{state_size}_space_{action_space_size}.csv")

    test = pd.read_csv(f"{base_directory}/q_tables/p_c_s_size_{state_size}_space_{action_space_size}.csv", index_col=0)

    print(q_values)
    print(test)

    print(q_values.loc["p_0.21_c_0.07_s_0",-1.0])
    print(test.loc["p_0.21_c_0.07_s_0", "-1.0"])

    test.to_csv(f"{base_directory}/q_tables/p_c_s_size_{state_size}_space_{action_space_size}.csv")

    test_2 = pd.read_csv(f"{base_directory}/q_tables/p_c_s_size_{state_size}_space_{action_space_size}.csv", index_col=0)

    print(test_2)
    print(test_2.loc["p_0.21_c_0.07_s_0", "-1.0"])

    for i in range(0, 5):
        q_values.to_csv(f"{base_directory}/q_tables/p_c_s_size_{state_size}_space_{action_space_size}_agent_id_{i}.csv")




if __name__ == '__main__':
    evaluate_data()