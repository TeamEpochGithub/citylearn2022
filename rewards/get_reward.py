from typing import List
import numpy as np

###########################################################################
#####                Specify your reward function here                #####
###########################################################################
from citylearn.cost_function import CostFunction


# def calculate_normalized_values(net_electricity_consumption_price, net_electricity_consumption_without_storage_price,
#                       net_electricity_consumption_emission, net_electricity_consumption_without_storage_emission,
#                       net_electricity_consumption, net_electricity_consumption_without_storage):
#     price_cost = CostFunction.price(net_electricity_consumption_price)[-1] / \
#                  CostFunction.price(net_electricity_consumption_without_storage_price)[-1]
#     emission_cost = CostFunction.carbon_emissions(net_electricity_consumption_emission)[-1] / \
#                     CostFunction.carbon_emissions(
#                         net_electricity_consumption_without_storage_emission)[-1]
#     ramping_cost = CostFunction.ramping(net_electricity_consumption)[-1] / \
#                    CostFunction.ramping(net_electricity_consumption_without_storage)[-1]
#     load_factor_cost = CostFunction.load_factor(net_electricity_consumption)[-1] / \
#                        CostFunction.load_factor(net_electricity_consumption_without_storage)[-1]
#     grid_cost = np.mean([ramping_cost, load_factor_cost])
#
#     return price_cost, emission_cost, grid_cost

def calculate_values(net_electricity_consumption_price,
                      net_electricity_consumption_emission,
                      net_electricity_consumption):
    price_cost = CostFunction.price(net_electricity_consumption_price)[-1]
    emission_cost = CostFunction.carbon_emissions(net_electricity_consumption_emission)[-1]
    ramping_cost = CostFunction.ramping(net_electricity_consumption)[-1]
    load_factor_cost = CostFunction.load_factor(net_electricity_consumption)[-1]
    grid_cost = np.mean([ramping_cost, load_factor_cost])

    return price_cost, emission_cost, grid_cost

def get_reward(electricity_consumption: List[float], carbon_emission: List[float], electricity_price: List[float],
               agent_ids: List[int]) -> List[float]:
    """CityLearn Challenge user reward calculation.

        Parameters
        ----------
        electricity_consumption: List[float]
            List of each building's/total district electricity consumption in [kWh].
        carbon_emission: List[float]
            List of each building's/total district carbon emissions in [kg_co2].
        electricity_price: List[float]
            List of each building's/total district electricity price in [$].
        agent_ids: List[int]
            List of agent IDs matching the ordering in `electricity_consumption`, `carbon_emission` and `electricity_price`.

        Returns
        -------
        rewards: List[float]
            Agent(s) reward(s) where the length of returned list is either = 1 (central agent controlling all buildings) 
            or = number of buildings (independent agent for each building).
        """

    # *********** BEGIN EDIT ***********
    carbon_emission = np.array(carbon_emission).clip(min=0)
    electricity_price = np.array(electricity_price).clip(min=0)
    reward = (carbon_emission + electricity_price) * -1
    return sum(reward) / 5
    # ************** END ***************

    # *********** BEGIN EDIT ***********
    ## net_electricity_consumption = `cooling_electricity_consumption` + `heating_electricity_consumption` + `dhw_electricity_consumption` + `electrical_storage_electricity_consumption` + `non_shiftable_load_demand` + `solar_generation`
    # district_electricity_price = electricity_price
    # district_electricity_consumption = electricity_consumption
    # district_carbon_emissions = carbon_emission
    # price_cost, emission_cost, grid_cost = calculate_values(district_electricity_price, district_carbon_emissions, district_electricity_consumption)
    # reward = ((price_cost + emission_cost + grid_cost)/5) * -1
    # return reward
    # ************** END ***************
