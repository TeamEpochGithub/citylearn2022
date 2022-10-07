import numpy as np
from numpy import sqrt


# Electrical storage consumption is -previous_soc*efficiency <= energy_normed <= (previous_capacity-previous_soc)/efficiency
# and if we take |action| <= nominal_power/previous_capacity, energy_normed = action*previous_capacity.

def find_efficiency(action, nominal_power, previous_capacity):

    x = np.abs(action * previous_capacity / nominal_power)

    if 0 <= np.abs(action) <= 0.3 * nominal_power / previous_capacity:
        efficiency = sqrt(0.83)
    elif 0.3 * nominal_power / previous_capacity < np.abs(action) < 0.7 * nominal_power / previous_capacity:
        efficiency = sqrt(0.7775 + 0.175 * x)
    elif 0.7 * nominal_power / previous_capacity <= np.abs(
            action) <= 0.8 * nominal_power / previous_capacity:  # Optimal efficiency
        efficiency = sqrt(0.9)
    elif 0.8 * nominal_power / previous_capacity < np.abs(action) <= nominal_power / previous_capacity:
        efficiency = sqrt(1.1 - 0.25 * x)

    return efficiency


def find_action_limit(action, nominal_power, previous_capacity, previous_soc):

    positive = 1 if action >= 0 else -1

    action = max(min(action, nominal_power / previous_capacity), -nominal_power / previous_capacity)
    energy = action * previous_capacity
    efficiency = find_efficiency(action, nominal_power, previous_capacity)

    actual_consumption = max(min(action, (previous_capacity-previous_soc)/efficiency), -1*previous_soc*efficiency)

    while not -1*previous_soc*efficiency <= energy <= (previous_capacity-previous_soc)/efficiency:

        action = action - positive*0.00001
        energy = action * previous_capacity

        new_efficiency = find_efficiency(action, nominal_power, previous_capacity)

        if efficiency == new_efficiency:
            if positive == -1:
                energy = -1 * previous_soc * efficiency
            else:
                energy = (previous_capacity - previous_soc) / efficiency

            efficiency = find_efficiency(energy/previous_capacity, nominal_power, previous_capacity)

        else:
            efficiency = new_efficiency

    return action, actual_consumption

