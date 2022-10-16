import numpy as np
from numpy import float_power as pw
from numpy import sqrt

def net_electricity_consumption(non_shiftable_load, electrical_storage_electricity_consumption, p_solar_generation_obs):
    # non_shiftable_load from csv data
    # solar_generation_obs from observation
    # electrical_storage_electricity_consumption is last_energy_balance
    return non_shiftable_load + electrical_storage_electricity_consumption - p_solar_generation_obs


def solar_generation_obs(solar_nominal_power, solar_generation_data):
    # solar_nominal_power[i] = list(env.get_building_information().values())[i]["solar_power"], for building i → (4, 4, 4, 5, 4)
    # solar_generation_data from csv data
    return solar_nominal_power*solar_generation_data/1000


def electrical_storage_soc(p_soc, previous_capacity):
    # soc is battery level in kWh for this time-step
    # previous_capacity is capacity of battery for the previous time-step
    # electrical_storage_soc is observation of battery level
    return p_soc/previous_capacity


def soc(p_energy_normed, p_soc_init, p_efficiency, previous_capacity):
    # energy normed is energy but cut-off at nominal level
    # soc_init is soc for previous time_step times (1-loss_coefficient)
    if p_energy_normed >= 0:
        p_soc = min(p_soc_init + p_energy_normed * p_efficiency, previous_capacity)
    else:
        p_soc = max(0, p_soc_init + p_energy_normed / p_efficiency)
    return p_soc


# def soc_init(previous_soc):
#     loss_coefficient = 0  # potentially 0.006?
#
#     return previous_soc*(1-loss_coefficient)


def energy_normed(p_energy, p_max_power):
    # energy is calculated using battery capacity and action
    if p_energy >= 0:
        return min(p_energy, p_max_power)
    else:
        return max(p_energy, -p_max_power)


# def max_power(p_soc_init, previous_capacity, nominal_power):
#     # nominal_power[i] = env.buildings[i].electrical_storage.nominal_power, 5.0 for all buildings
#     capacity_power_curve = [[0.0, 1], [0.8, 1], [1.0, 0.2]]
#     capacity_power_curve = np.array(capacity_power_curve).T
#
#     soc_normalized = p_soc_init/previous_capacity
#     idx = max(0, np.argmax(soc_normalized <= capacity_power_curve[0]) - 1)
#     max_output_power = nominal_power * (
#             capacity_power_curve[1][idx]
#             + (capacity_power_curve[1][idx + 1] - capacity_power_curve[1][idx]) * (
#                         soc_normalized - capacity_power_curve[0][idx])
#             / (capacity_power_curve[0][idx + 1] - capacity_power_curve[0][idx])
#     )
#
#     return max_output_power


def max_power(p_soc_init, p_nominal_power, previous_capacity):

    x = p_soc_init / previous_capacity

    if 0 <= x <= 0.8:
        p_max_power = p_nominal_power
    elif 0.8 < x <= 1:
        p_max_power = p_nominal_power * (4 - 4 * x)

    return p_max_power


def energy(p_action, previous_capacity):
    # capacity is current battery capacity in kWh
    return p_action*previous_capacity


def new_capacity(previous_capacity, p_last_energy_balance):
    capacity_loss_coefficient = pw(10.0, -5)
    initial_capacity = 6.4
    capacity_degrade = capacity_loss_coefficient * initial_capacity * np.abs(p_last_energy_balance) / (2 * previous_capacity)
    return previous_capacity - capacity_degrade


def last_energy_balance(p_soc, previous_soc, p_efficiency):
    loss_coefficient = 0  # potentially 0.006?
    energy_balance = p_soc - previous_soc * (1 - loss_coefficient)
    if energy_balance >= 0:
        energy_balance = energy_balance / p_efficiency
    else:
        energy_balance = energy_balance * p_efficiency

    return energy_balance


# def efficiency(p_energy_normed, nominal_power):
#     # nominal_power[i] = env.buildings[i].electrical_storage.nominal_power, 5.0 for all buildings
#     power_efficiency_curve = [[0, 0.83],[0.3, 0.83],[0.7, 0.9],[0.8, 0.9],[1, 0.85]]
#     power_efficiency_curve = np.array(power_efficiency_curve).T
#     efficiency_scaling = 0.5
#
#     energy_normalized = np.abs(p_energy_normed) / nominal_power
#     idx = max(0, np.argmax(energy_normalized <= power_efficiency_curve[0]) - 1)
#     get_efficiency = power_efficiency_curve[1][idx] \
#                  + (energy_normalized - power_efficiency_curve[0][idx]
#                     ) * (power_efficiency_curve[1][idx + 1] - power_efficiency_curve[1][idx]
#                          ) / (power_efficiency_curve[0][idx + 1] - power_efficiency_curve[0][idx])
#     get_efficiency = get_efficiency ** efficiency_scaling
#
#     return get_efficiency

def efficiency(p_energy_normed, p_nominal_power):

    x = np.abs(p_energy_normed / p_nominal_power)

    if 0 <= x <= 0.3:
        p_efficiency = sqrt(0.83)
    elif 0.3 < x < 0.7:
        p_efficiency = sqrt(0.7775 + 0.175 * x)
    elif 0.7 <= x <= 0.8:  # Optimal efficiency
        p_efficiency = sqrt(0.9)
    elif 0.8 < x <= 1:
        p_efficiency = sqrt(1.1 - 0.25 * x)

    return p_efficiency


    # # Example for second observation, taking action 0.5:
    #
    # v_action = 0.5
    #
    # v_previous_capacity = 6.4
    # v_previous_soc = 0
    # v_nominal_power = 5.0
    #
    # v_max_power = max_power(p_soc_init = v_previous_soc, p_nominal_power = v_nominal_power, previous_capacity = v_previous_capacity)
    #
    # v_energy_normed = energy_normed(p_energy=energy(action=v_action, previous_capacity=v_previous_capacity), p_max_power = v_max_power)
    #
    # v_efficiency = efficiency(energy_normed=energy_normed, nominal_power=v_nominal_power)
    #
    # v_soc = soc(energy_normed=energy_normed, soc_init=v_previous_soc, efficiency=efficiency, previous_capacity=previous_capacity)
    #
    # v_last_energy_balance = last_energy_balance(soc=v_soc, previous_soc=v_previous_soc, efficiency=v_efficiency)
    #
    # v_capacity = new_capacity(previous_capacity=previous_capacity, last_energy_balance=v_last_energy_balance)
    #
    # battery = electrical_storage_soc(soc=v_soc, previous_capacity=previous_capacity)
    #
    # # print(f"Battery observation: {battery}")
    #
    # # Correct result should be: 0.47156653825308686
    #
    #
    # # For building 1:
    #
    # solar_nominal_power = 4
    # solar_generation_data = 0
    # non_shiftable_load = 0.8511666666666671
    #
    #
    # v_solar_generation_obs = solar_generation_obs(solar_nominal_power=solar_nominal_power, solar_generation_data=solar_generation_data)
    #
    # v_net_electricity_consumption = net_electricity_consumption(non_shiftable_load, v_last_energy_balance, solar_generation_obs)
    #
    # # Correct result: 4.051166666666667


