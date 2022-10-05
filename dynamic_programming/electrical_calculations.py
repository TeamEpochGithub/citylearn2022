import numpy as np


def net_electricity_consumption(non_shiftable_load, electrical_storage_electricity_consumption, solar_generation_obs):
    # non_shiftable_load from csv data
    # solar_generation_obs from observation
    # electrical_storage_electricity_consumption is last_energy_balance
    return non_shiftable_load + electrical_storage_electricity_consumption - solar_generation_obs


def solar_generation_obs(nominal_power, solar_generation_data):
    # nominal_power[i] = list(env.get_building_information().values())[i]["solar_power"], for building i → (4, 4, 4, 5, 4)
    # solar_generation_data from csv data
    return nominal_power * solar_generation_data / 1000


def electrical_storage_soc(soc, previous_capacity):
    # soc is battery level in kWh for this time-step
    # previous_capacity is capacity of battery for the previous time-step
    return soc / previous_capacity


def soc(energy_normed, soc_init, efficiency, previous_capacity):
    # energy normed is energy but cut-off at nominal level
    # soc_init is soc for previous time_step times (1-loss_coefficient)
    if energy_normed >= 0:
        soc = min(soc_init + energy_normed * efficiency, previous_capacity)
    else:
        soc = max(0, soc_init + energy_normed / efficiency)
    return soc


def soc_init(previous_soc):
    loss_coefficient = 0  # potentially 0.006?

    return previous_soc * (1 - loss_coefficient)


def energy_normed(energy):
    # energy is calculated using battery capacity and action
    if energy >= 0:
        return min(energy, 5)
    else:
        return max(energy, -5)


def energy(action, previous_capacity):
    # capacity is current battery capacity in kWh
    return action * previous_capacity


def new_capacity(previous_capacity, last_energy_balance):
    capacity_loss_coefficient = np.power(10.0, -5)
    initial_capacity = 6.4
    capacity_degrade = capacity_loss_coefficient * initial_capacity * np.abs(last_energy_balance) / (
                2 * previous_capacity)
    return previous_capacity - capacity_degrade


def last_energy_balance(soc, previous_soc, efficiency):
    loss_coefficient = 0  # potentially 0.006?
    energy_balance = soc - previous_soc * (1 - loss_coefficient)
    if energy_balance >= 0:
        energy_balance = energy_balance / efficiency
    else:
        energy_balance = energy_balance * efficiency

    return energy_balance


def efficiency(energy_normed):
    nominal_power = 5.0
    power_efficiency_curve = [[0, 0.83], [0.3, 0.83], [0.7, 0.9], [0.8, 0.9], [1, 0.85]]
    power_efficiency_curve = np.array(power_efficiency_curve).T
    efficiency_scaling = 0.5

    energy_normalized = np.abs(energy_normed) / nominal_power
    idx = max(0, np.argmax(energy_normalized <= power_efficiency_curve[0]) - 1)
    get_efficiency = power_efficiency_curve[1][idx] \
                     + (energy_normalized - power_efficiency_curve[0][idx]
                        ) * (power_efficiency_curve[1][idx + 1] - power_efficiency_curve[1][idx]
                             ) / (power_efficiency_curve[0][idx + 1] - power_efficiency_curve[0][idx])
    get_efficiency = get_efficiency ** efficiency_scaling

    return get_efficiency


def calculate_net_consumption_soc_capacity(action, previous_capacity, previous_soc, non_shiftable_load, solar_generation_data, nominal_power):

    energy_normed_value = energy_normed(energy(action, previous_capacity))

    efficiency_value = efficiency(energy_normed_value)

    soc_value = soc(energy_normed_value, previous_soc, efficiency_value, previous_capacity)

    last_energy_balance_value = last_energy_balance(soc_value, previous_soc, efficiency_value)

    new_capacity_value = new_capacity(previous_capacity, last_energy_balance_value)

    solar_generation_value = solar_generation_obs(nominal_power, solar_generation_data)

    net_electricity_consumption_value = net_electricity_consumption(non_shiftable_load, last_energy_balance_value, solar_generation_value)

    return net_electricity_consumption_value, soc_value, new_capacity_value

