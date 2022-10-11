import numpy as np
from traineval.training.data_preprocessing.net_electricity_consumption import net_electricity_consumption, non_shiftable_load, v_last_energy_balance, solar_generation_obs

def price_cost(net_electricity_consumption_prices, net_electricity_consumption_prices_wout_storage):
    consumption_price = sum(net_electricity_consumption_prices)
    consumption_price_wout_storage = sum(net_electricity_consumption_prices_wout_storage) #8277.733108897906
    #Adding the money spent per hour on all buildings, for all hours

    return consumption_price/consumption_price_wout_storage

def emission_cost(net_electricity_consumption_emissions, net_electricity_consumption_emissions_wout_storage):
    return price_cost(net_electricity_consumption_emissions, net_electricity_consumption_emissions_wout_storage) #4627.487397661689
    #Adding the emission per hour for all buildings, for all hours

def net_electricity_consumption_prices(net_building_hourly_electricity_consumptions, elecitricity_pricing):
    consumption_prices = []

    for hour in len(net_building_hourly_electricity_consumptions):
        consumption_prices.append(max(0, sum([b*elecitricity_pricing[hour] for b in net_building_hourly_electricity_consumptions[hour]])))
        #Money spent each hour on all buildings combined
        #Building sum of hourly electricity consumption price is >= 0!

    return consumption_prices

def net_electricity_consumption_emissions(net_building_hourly_electricity_consumptions, carbon_emission):
    consumption_emissions = []

    for hour in len(net_building_hourly_electricity_consumptions):
        consumption_emissions.append(sum([max(0, b)*carbon_emission[hour] for b in net_building_hourly_electricity_consumptions[hour]]))
        #Carbon emission of all buildings combined each hour
        #Individual building hourly electricity emission is >= 0!

    return consumption_emissions

def net_building_hourly_electricity_consumptions():
    net_building_hourly_electricity_consumptions_w_storage = []
    net_building_hourly_electricity_consumptions_wout_storage = []

    for hour in range(8760):

        buildings = []
        buildings_wout_storage = []

        for i in range(5):
            buildings.append(net_electricity_consumption(non_shiftable_load, v_last_energy_balance, solar_generation_obs))
            buildings_wout_storage.append(net_electricity_consumption(non_shiftable_load, 0, solar_generation_obs))
            # Introduce correct values
        #List of individual net_electricity_consumption for each building in a single hour

        net_building_hourly_electricity_consumptions_w_storage.append(buildings)
        net_building_hourly_electricity_consumptions_wout_storage.append(buildings_wout_storage)

        #List of lists of building net_electricity_consumption for all hours

    return [net_building_hourly_electricity_consumptions_w_storage, net_building_hourly_electricity_consumptions_wout_storage]

def ramping_cost(net_building_hourly_electricity_consumptions):
    net_consumptions = [sum(hour) for hour in net_building_hourly_electricity_consumptions[0]]
    net_consumptions_wout_storage = [sum(hour) for hour in net_building_hourly_electricity_consumptions[1]]

    ramping = []
    ramping_wout_storage = []

    for i in range(1, len(net_consumptions)):
        ramping.append(abs(net_consumptions[i]-net_consumptions[i-1]))
        ramping_wout_storage.append(abs(net_consumptions_wout_storage[i]-net_consumptions_wout_storage[i-1]))
    #Hourly differences in total district net_electricity_consumption

    ramping = sum(ramping)
    ramping_wout_storage = sum(ramping_wout_storage) #14807.707606563112

    return ramping/ramping_wout_storage

def load_factor_cost(net_building_hourly_electricity_consumptions):
    net_consumptions = [sum(hour) for hour in net_building_hourly_electricity_consumptions[0]]
    net_consumptions_wout_storage = [sum(hour) for hour in net_building_hourly_electricity_consumptions[1]]
    load_factor_costs = []

    for i in [net_consumptions, net_consumptions_wout_storage]:
        load_factors = []

        for month in range(12):
            consumptions = np.array(i[month*730:month*730+730])

            mean = np.mean(consumptions)
            vmax = np.amax(consumptions)
            load_factor = 1-(mean/vmax)

            load_factors.append(load_factor)

        load_factor_costs.append(np.array(load_factors).mean())

    return load_factor_costs[0]/load_factor_costs[1] #0.869759047619068

def grid_cost(ramping_cost, load_factor_cost):
    return np.mean([ramping_cost, load_factor_cost])

def cost(price_cost, emission_cost, grid_cost):
    return np.mean([price_cost, emission_cost, grid_cost])
