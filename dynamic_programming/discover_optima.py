# Select an hour as entry point
# Split the current action space and recursively explore them
# At each recursive step, the hour is incremented and the action space of that hour is split.
# When the required depth is reached, the current hour is evaluated and the results are pushed upwards.
# At each split, the best scoring action and its evaluation are added to a list.
# When this process terminates, the list will indicate which action at the entry point is optimal.
# Further runs of this process will split the action spaces into increasingly smaller regions.

# net_electricity_consumption_without_storage = non_shiftable_load_demand + solar_generation
# net_electricity_consumption = electrical_storage_electricity_consumption + non_shiftable_load_demand + solar_generation

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


def price(net_electricity_consumption_price_list):
    net_electricity_consumption_price_list = np.asarray(net_electricity_consumption_price_list)
    return np.cumsum(np.clip(net_electricity_consumption_price_list, a_min=0, a_max=None))[-1]

def carbon(net_electricity_consumption_carbon_list):
    net_electricity_consumption_carbon_list = np.asarray(net_electricity_consumption_carbon_list)
    return np.cumsum(np.clip(net_electricity_consumption_carbon_list, a_min=0, a_max=None))[-1]

def ramping(net_electricity_consumption):

    pass