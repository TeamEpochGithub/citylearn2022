import numpy as np

def net_electricity_consumption(non_shiftable_load, electrical_storage_electricity_consumption, solar_generation_obs):
    #non_shiftable_load from csv data
    #solar_generation_obs from observation
    #electrical_storage_electricity_consumption is last_energy_balance
    return non_shiftable_load + electrical_storage_electricity_consumption - solar_generation_obs


def solar_generation_obs(nominal_power, solar_generation_data):
    #nominal_power[i] = list(env.get_building_information().values())[i]["solar_power"], for building i → (4, 4, 4, 5, 4)
    #solar_generation_data from csv data
    return nominal_power*solar_generation_data/1000


def electrical_storage_soc(soc, previous_capacity):
    #soc is battery level in kWh for this time-step
    #previous_capacity is capacity of battery for the previous time-step
    #electrical_storage_soc is observation of battery level
    return soc/previous_capacity


def soc(energy_normed, soc_init, efficiency, previous_capacity):
    #energy normed is energy but cut-off at nominal level
    #soc_init is soc for previous time_step times (1-loss_coefficient)
    if energy_normed >= 0:
        soc = min(soc_init + energy_normed*efficiency, previous_capacity)
    else:
        soc = max(0, soc_init + energy_normed/efficiency)
    return soc

def soc_init(previous_soc):
    loss_coefficient = 0  # potentially 0.006?

    return previous_soc*(1-loss_coefficient)


def energy_normed(energy):
    #energy is calculated using battery capacity and action
    if energy >= 0:
        return min(energy, 5)
    else:
        return max(energy, -5)


def energy(action, previous_capacity):
    #capacity is current battery capacity in kWh
    return action*previous_capacity


def new_capacity(previous_capacity, last_energy_balance):
    capacity_loss_coefficient = np.power(10.0, -5)
    initial_capacity = 6.4
    capacity_degrade = capacity_loss_coefficient*initial_capacity*np.abs(last_energy_balance)/(2*previous_capacity)
    return previous_capacity - capacity_degrade


def last_energy_balance(soc, previous_soc, efficiency):
    loss_coefficient = 0  #potentially 0.006?
    energy_balance = soc - previous_soc * (1-loss_coefficient)
    if energy_balance >= 0:
        energy_balance = energy_balance/efficiency
    else:
        energy_balance = energy_balance*efficiency

    return energy_balance


def efficiency(energy_normed):
    nominal_power = 5.0
    power_efficiency_curve = [[0, 0.83],[0.3, 0.83],[0.7, 0.9],[0.8, 0.9],[1, 0.85]]
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


#Example for second observation, taking action 0.5:

action = 0.638

previous_capacity = 6.4
previous_soc = 0


# Electrical storage consumption is -previous_soc*efficiency <= energy_normed <= (previous_capacity-previous_soc)/efficiency
# and if we take |action| <= 5/previous_capacity, energy_normed = action*previous_capacity.
# We can take -sqrt(0.83)*previous_soc <= action*previous_capacity <= (previous_capacity-previous_soc)/sqrt(0.9) and we are safe


# if action >= 0:
#     action = min(5/previous_capacity, action)
# else:
#     action = max(-5/previous_capacity, action)

energy_normed = energy_normed(energy=energy(action=action, previous_capacity=previous_capacity))

efficiency = efficiency(energy_normed=energy_normed)

# if energy_normed >= 0:
#     consumption = min((previous_capacity-previous_soc)/efficiency, energy_normed)
# else:
#     consumption = max(-1*previous_soc, energy_normed)

soc = soc(energy_normed=energy_normed, soc_init=soc_init(previous_soc=previous_soc), efficiency=efficiency, previous_capacity=previous_capacity)

last_energy_balance = last_energy_balance(soc=soc, previous_soc=previous_soc, efficiency=efficiency)

capacity = new_capacity(previous_capacity=previous_capacity, last_energy_balance=last_energy_balance)

battery = electrical_storage_soc(soc=soc, previous_capacity=previous_capacity)

print(f"Battery observation: {battery}")

#Correct result should be: 0.47156653825308686


#For building 1:

solar_generation_obs = solar_generation_obs(nominal_power=4, solar_generation_data=0)

non_shiftable_load = 0.8511666666666671

net_electricity_consumption = net_electricity_consumption(non_shiftable_load, last_energy_balance, solar_generation_obs)

#Correct result: 4.051166666666667

