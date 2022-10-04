# Select an hour as entry point
# Split the current action space and recursively explore them
# At each recursive step, the hour is incremented and the action space of that hour is split.
# When the required depth is reached, the current hour is evaluated and the results are pushed upwards.
# At each split, the best scoring action and its evaluation are added to a list.
# When this process terminates, the list will indicate which action at the entry point is optimal.
# Further runs of this process will split the action spaces into increasingly smaller regions.

# net_electricity_consumption_without_storage = non_shiftable_load_demand - solar_generation
# net_electricity_consumption = electrical_storage_electricity_consumption + non_shiftable_load_demand - solar_generation

# net_electricity_consumption_price = net_electricity_consumption * pricing.electricity_pricing
# net_electricity_consumption_price_without_storage = net_electricity_consumption_without_storage * pricing.electricity_pricing

# net_electricity_consumption_emission = max(0, net_electricity_consumption * carbon_intensity.carbon_intensity)
# net_electricity_consumption_emission_without_storage = max(0, net_electricity_consumption_without_storage * carbon_intensity.carbon_intensity)

# values here are saved as sums, with the most recent value added used to calculate the score. (rolling window sum)

# In the building.csv files, non-shiftable-load is called "Equipment Electric Power"

# Price function: For net_electricity_consumption_price and net_electricity_consumption_price_without_storage calculate a rolling sum
#               : Divide the normal sum by the non-battery storage sum

# Emission function: For net_electricity_consumption_emission and net_electricity_consumption_emission_without_storage calculate a rolling sum
#               : Divide the normal sum by the non-battery storage sum

# Ramping function: (list of net_electricity_consumption values) - (list list of net_electricity_consumption values shifted by 1)
#                 : make all values in list absolute
#                 : take a rolling sum of the list
#                 : take final value of the list
#                 : apply this process to net_electricity_consumption list and net_electricity_consumption_without_storage list
#                 : divide the value using battery storage by the value not using battery storage

# Load Factor function: Divide the list of net_electricity_consumption values into sections of 730 values each
#                     : For each section calculate the mean and max values
#                     : For each section, divide the mean by the max, and subtract the result from 1
#                     : Take a rolling mean of these values
#                     : Take the final element in the list
#                     : Apply this process net_electricity_consumption list and net_electricity_consumption_without_storage
#                     : Divide the battery storage value by the non-battery storage value
#
# Grid cost is the mean of Load Factor and Ramping
import numpy as np
from dynamic_programming import electrical_calculations


def evaluate_observation(observation_list):

    newlist = [] * 5
    for observation in observation_list:
        for ind, building in enumerate(observation):
            newlist[ind].append(building)

    scores = []
    for building_obs in newlist:
        observation_list = np.asarray(building_obs)

        net_electricity_consumption = observation_list[:, 23]
        non_shiftable_load = observation_list[:, 20]
        electricity_pricing = observation_list[:, 24]
        carbon_intensity = observation_list[:, 19]
        solar_generation = observation_list[:, 21]

        net_electricity_consumption_without_storage = non_shiftable_load - solar_generation

        scores.append(np.asarray(
            evaluate(net_electricity_consumption, net_electricity_consumption_without_storage, electricity_pricing,
                     carbon_intensity)))

    scores = np.asarray(scores)
    return np.mean(scores[:, 0]), np.mean(scores[:, 1]), np.mean(scores[:, 2])


def evaluate(net_electricity_consumption, net_electricity_consumption_without_storage, electricity_pricing,
             carbon_intensity):
    net_electricity_consumption = np.asarray(net_electricity_consumption)
    net_electricity_consumption_without_storage = np.asarray(net_electricity_consumption_without_storage)
    electricity_pricing = np.asarray(electricity_pricing)
    carbon_intensity = np.asarray(carbon_intensity)

    net_electricity_consumption_price = net_electricity_consumption * electricity_pricing
    net_electricity_consumption_price_without_storage = net_electricity_consumption_without_storage * electricity_pricing

    net_electricity_consumption_emission = net_electricity_consumption * carbon_intensity
    net_electricity_consumption_emission_without_storage = net_electricity_consumption_without_storage * carbon_intensity

    price_score = price(net_electricity_consumption_price) / price(net_electricity_consumption_price_without_storage)
    carbon_score = carbon(net_electricity_consumption_emission) / carbon(
        net_electricity_consumption_emission_without_storage)

    load_factor_score = load_factor(net_electricity_consumption) / load_factor(
        net_electricity_consumption_without_storage)
    ramping_score = ramping(net_electricity_consumption) / ramping(net_electricity_consumption_without_storage)

    grid_score = (load_factor_score + ramping_score) / 2

    return price_score, carbon_score, grid_score


def price(net_electricity_consumption_price_list):
    net_electricity_consumption_price_list = np.asarray(net_electricity_consumption_price_list)
    return np.cumsum(np.clip(net_electricity_consumption_price_list, a_min=0, a_max=None))[-1]


def carbon(net_electricity_consumption_carbon_list):
    net_electricity_consumption_carbon_list = np.asarray(net_electricity_consumption_carbon_list)
    return np.cumsum(np.clip(net_electricity_consumption_carbon_list, a_min=0, a_max=None))[-1]


def ramping(net_electricity_consumption):
    net_electricity_consumption = np.asarray(net_electricity_consumption)

    shifted_net_electricity_consumption = np.roll(net_electricity_consumption, 1)
    shifted_net_electricity_consumption[0] = 0

    subtracted_net_electricity_consumption = np.abs(net_electricity_consumption - shifted_net_electricity_consumption)
    subtracted_net_electricity_consumption[0] = 0

    return np.cumsum(subtracted_net_electricity_consumption)[-1]


def load_factor(net_electricity_consumption):
    net_electricity_consumption = np.asarray(net_electricity_consumption)

    window_size = 730

    windowed_net_electricity_consumption = np.split(net_electricity_consumption,
                                                    np.arange(window_size, len(net_electricity_consumption),
                                                              window_size))

    calculation_list = []

    for window in windowed_net_electricity_consumption:
        window_max = window.max()
        window_mean = window.mean()

        calculation_list.append(1 - window_mean / window_max)

    return np.asarray(calculation_list).mean()
