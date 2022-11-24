import numpy as np
import os.path as osp
import csv

from agents.helper_classes.load_learner_instances import LoadLag0Learner, LoadLag1Learner, LoadLag2Learner, \
    LoadLag3Learner, \
    LoadLag4Learner, \
    LoadLag5Learner, LoadLag6Learner
from agents.helper_classes.solar_learner_instances import SolarLag0Learner, SolarLag2Learner, SolarLag3Learner, \
    SolarLag4Learner, SolarLag5Learner, SolarLag6Learner, SolarLag1Learner

from analysis import analysis_data
from agents.helper_classes.live_learning import LiveLearner
from traineval.training.data_preprocessing import net_electricity_consumption as n
from traineval.training.data_preprocessing.pricing_simplified import pricing, shift_date


# consumptions_path = osp.join(osp.dirname(competition_data.__file__), "consumptions/building_consumptions.csv")
# carbon_path = osp.join(osp.dirname(competition_data.__file__), "carbon_intensity.csv")

# consumptions = pd.read_csv(consumptions_path)[[f"{i}" for i in range(5)]]
# consumptions = [consumptions[f"{i}"].values.tolist()[1:] for i in range(5)]

# carbon = pd.read_csv(carbon_path)["kg_CO2/kWh"]
# carbon = carbon.values.tolist()[1:]

def write_step_to_file(agent_id, timestep, action, observation):
    # ID, Action, Battery level, Consumption, Load, Solar, Carbon, Price
    row = [agent_id, timestep, action, observation[22], observation[23], observation[20], observation[21],
           observation[19],
           observation[24]]
    action_file_path = osp.join(osp.dirname(analysis_data.__file__), 'pred_performance.csv')
    action_file = open(action_file_path, 'a', newline="")
    writer = csv.writer(action_file)
    writer.writerow(row)
    action_file.close()


def get_district_chunk_consumptions_fit_delay(consumption_sign, load_learners, solar_learners, timestep):
    max_chunk_size = 32

    if timestep + max_chunk_size > 8759:
        max_chunk_size = 8759 - timestep

    num_buildings = len(load_learners)
    all_chunk_consumptions = []

    for learner in range(num_buildings):
        loads = load_learners[learner].predict_load(max_chunk_size)
        solars = solar_learners[learner].predict_solar(max_chunk_size)

        all_chunk_consumptions.append([a - b for a, b in zip(loads, solars)])

    district_chunk_consumptions = [sum([all_chunk_consumptions[b][c] for b in range(num_buildings)]) for c in
                                   range(max_chunk_size)]

    for index, consumption in enumerate(district_chunk_consumptions):
        if consumption * consumption_sign < 0:
            district_chunk_consumptions = district_chunk_consumptions[:index]
            break

    return district_chunk_consumptions


def extra_charge(remaining_battery_capacity, soc, chunk_consumptions, chunk_charge_loads_in, date):
    chunk_total_consumption = sum(chunk_consumptions)
    chunk_charge_loads = chunk_charge_loads_in

    remaining_possible_charge = (remaining_battery_capacity - soc) / np.sqrt(0.83) + chunk_total_consumption

    consumption_prices, prices = get_consumption_prices(date, chunk_consumptions)

    price_occurrences = list(set(prices))
    price_indexes = [[i for i, p in enumerate(prices) if p == p_occurrence] for p_occurrence in
                     price_occurrences]

    if len(price_indexes) == 2:

        for i, price_occurrence_indexes in enumerate(price_indexes):

            if i == 0:
                opposite_index = 1
            elif i == 1:
                opposite_index = 0

            for price_index in price_occurrence_indexes:
                chunk_charge_loads[price_index] += remaining_possible_charge / \
                                                   (len(price_occurrence_indexes) +
                                                    len(price_indexes[opposite_index]) *
                                                    (price_occurrences[i] / price_occurrences[opposite_index]))

    else:
        chunk_charge_loads = [c + remaining_possible_charge / len(chunk_charge_loads_in) for c in
                              chunk_charge_loads_in]

    return chunk_charge_loads


def negative_consumption_scenario(date, chunk_consumptions, remaining_battery_capacity, soc):
    chunk_total_consumption = sum(chunk_consumptions)

    if -1 * chunk_total_consumption >= (remaining_battery_capacity - soc) / np.sqrt(0.83):
        # If more energy can be obtained than the one necessary to charge the battery
        relative_consumption = [i / chunk_total_consumption for i in chunk_consumptions]
        chunk_charge_loads = [i * (remaining_battery_capacity - soc) / np.sqrt(0.83) for i in relative_consumption]

    else:  # Otherwise charge with all the possible energy
        chunk_charge_loads = [-1 * i for i in chunk_consumptions]

        if -chunk_total_consumption >= 0.25 * ((remaining_battery_capacity - soc) / np.sqrt(0.83)):
            chunk_charge_loads = extra_charge(remaining_battery_capacity, soc, chunk_consumptions, chunk_charge_loads,
                                              date)

    return chunk_charge_loads


def lowering_peaks(local_soc, chunk_charge_loads, consumption_prices, prices):
    while local_soc != 0:

        # Get the peak consumption_price and check in which step the peak(s) happens
        max_consumption_price = max(consumption_prices)
        peak_indices = [i for i, p in enumerate(consumption_prices) if p == max_consumption_price]

        # List of other prices which do not indicate a peak
        consumption_prices_without_peak = [x for x in consumption_prices if x != max_consumption_price]

        if len(consumption_prices_without_peak) == 0:
            consumption_prices_without_peak = [0]

        # Get the difference in consumption price between the highest peak and the next highest peak
        # Make a list of the differences in consumption between the highest peaks and the next highest peak
        difference_from_peak = max_consumption_price - max(consumption_prices_without_peak)
        consumption_difference = [difference_from_peak / prices[i] for i in peak_indices]

        # Lower peaks to next highest peak
        if local_soc >= sum(consumption_difference):
            for i, difference in enumerate(consumption_difference):
                chunk_charge_loads[peak_indices[i]] += difference
                local_soc -= difference
                consumption_prices[peak_indices[i]] -= difference_from_peak
        else:
            relative_difference = [c / sum(consumption_difference) for c in consumption_difference]

            for i, rd in enumerate(relative_difference):
                chunk_charge_loads[peak_indices[i]] += local_soc * rd
                consumption_prices[peak_indices[i]] -= rd * local_soc * prices[peak_indices[i]]

            local_soc = 0

    for i in range(2, len(chunk_charge_loads)):
        if chunk_charge_loads[i] != 0 and chunk_charge_loads[i - 1] == 0 and chunk_charge_loads[i - 2] == 0:
            chunk_charge_loads[i - 2] = 0.000000001
            chunk_charge_loads[i - 1] = -0.0000001
            break

    return chunk_charge_loads


def get_consumption_prices(obs_date, chunk_consumptions):
    date = obs_date
    prices = []

    for hour in range(len(chunk_consumptions)):
        prices.append(pricing(date[2], date[0], date[1]))
        date = shift_date(date[0], date[1], date[2], shifts=1)

    consumption_prices = [prices[i] * c for i, c in enumerate(chunk_consumptions)]

    return consumption_prices, prices


def positive_consumption_scenario(date, chunk_consumptions, soc):
    chunk_total_consumption = sum(chunk_consumptions)

    if chunk_total_consumption >= soc * np.sqrt(0.83):
        # If fully discharging the battery doesn't bring the consumption to 0, we take the highest
        # price*consumption value and bring it down to the next highest price*consumption by reducing the
        # consumption at that time step. We do this consecutively until the battery has been emptied.

        consumption_prices, prices = get_consumption_prices(date, chunk_consumptions)

        local_soc = soc * np.sqrt(0.83)
        chunk_charge_loads = [0] * len(chunk_consumptions)

        return lowering_peaks(local_soc, chunk_charge_loads, consumption_prices, prices)

    else:
        return chunk_consumptions


def calculate_next_chunk(observation, consumption_sign, agent_id, timestep, remaining_battery_capacity, soc,
                         load_learners, solar_learners, stored_district_consumptions):
    if agent_id == 0:
        chunk_consumptions = get_district_chunk_consumptions_fit_delay(consumption_sign, load_learners, solar_learners,
                                                                       timestep)
    else:
        chunk_consumptions = stored_district_consumptions

    date = shift_date(observation[2], observation[1], observation[0], shifts=1)

    if consumption_sign == -1:  # If negative consumption
        chunk_charge_loads = negative_consumption_scenario(date, chunk_consumptions, remaining_battery_capacity, soc)
    else:
        chunk_charge_loads = positive_consumption_scenario(date, chunk_consumptions, soc)

    return chunk_charge_loads, chunk_consumptions


def day_night_policy(hour):
    action = -0.067
    if 6 <= hour <= 14:
        action = 0.11
    return action


class DistrictPredConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.remaining_battery_capacity = {}
        self.soc = {}
        self.write_to_file = False

        self.load_learner_options = {}
        self.load_learner_best_ind = {}
        self.solar_learner_options = {}
        self.solar_learner_best_ind = {}

        self.evaluation_timesteps = [500, 2000, 5000, -10]  # -10 added to prevent 'index out of range'
        self.evaluation_ind = 0
        self.evaluation_period = 168
        self.evaluation_left_bound = -10
        self.evaluation_right_bound = -10

        self.stored_district_consumptions = []

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.remaining_battery_capacity[agent_id] = 6.4
        self.soc[agent_id] = 0

        if str(agent_id) not in self.load_learner_best_ind:
            self.load_learner_best_ind[str(agent_id)] = 0

        if str(agent_id) not in self.load_learner_options:
            self.load_learner_options[str(agent_id)] = [LoadLag0Learner(500, 15),
                                                        LoadLag1Learner(500, 15),
                                                        LoadLag2Learner(500, 15),
                                                        LoadLag3Learner(500, 15),
                                                        LoadLag4Learner(500, 15),
                                                        LoadLag5Learner(500, 15),
                                                        LoadLag6Learner(500, 15)
                                                        ]

        if str(agent_id) not in self.solar_learner_best_ind:
            self.solar_learner_best_ind[str(agent_id)] = 0

        if str(agent_id) not in self.solar_learner_options:
            self.solar_learner_options[str(agent_id)] = [SolarLag0Learner(500, 15),
                                                         SolarLag1Learner(500, 15),
                                                         SolarLag2Learner(500, 15),
                                                         SolarLag3Learner(500, 15),
                                                         SolarLag4Learner(500, 15),
                                                         SolarLag5Learner(500, 15),
                                                         SolarLag6Learner(500, 15)
                                                         ]

    def compute_action(self, district_observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        building_timestep = self.timestep // len(district_observation)
        observation = district_observation[agent_id]

        action_out = \
            self.pred_consumption_policy(district_observation, building_timestep, agent_id,
                                         self.remaining_battery_capacity[agent_id],
                                         self.soc[agent_id], self.write_to_file)

        action = float(np.array(action_out, dtype=self.action_space[agent_id].dtype))
        max_power = n.max_power(self.soc[agent_id], 5, self.remaining_battery_capacity[agent_id])
        energy = n.energy_normed(action * self.remaining_battery_capacity[agent_id], max_power)
        efficiency = n.efficiency(energy, 5)

        previous_soc = self.soc[agent_id]
        self.soc[agent_id] = n.soc(energy, previous_soc, efficiency, self.remaining_battery_capacity[agent_id])

        battery_cons = n.last_energy_balance(self.soc[agent_id], previous_soc, efficiency)
        self.remaining_battery_capacity[agent_id] = n.new_capacity(self.remaining_battery_capacity[agent_id],
                                                                   battery_cons)

        return np.array([action_out], dtype=self.action_space[agent_id].dtype)

    def pred_consumption_policy(self, district_observation, timestep, agent_id, remaining_battery_capacity, soc,
                                write_to_file):
        if timestep >= 8758:
            return -1

        print("timestep: ", timestep)

        num_buildings = len(district_observation)
        observation = district_observation[agent_id]

        if agent_id == 0:
            for agent in range(num_buildings):
                self.set_evaluation_bounds(timestep)
                self.update_load_forecasters(agent, timestep, district_observation[agent][20])
                self.update_solar_forecasters(agent, timestep, district_observation[agent][21])

        if timestep < 72:
            return day_night_policy(observation[2])

        load_learners = []
        solar_learners = []

        if agent_id == 0:
            next_district_consumption = 0

            for agent in range(num_buildings):

                if self.evaluation_left_bound <= timestep <= self.evaluation_right_bound:
                    self.evaluate_learners(timestep, district_observation[agent][20], district_observation[agent][21], agent)

                load_learner = self.load_learner_options[str(agent)][self.load_learner_best_ind[str(agent)]]
                next_load = load_learner.predict_load(1)[0]
                solar_learner = self.solar_learner_options[str(agent)][self.solar_learner_best_ind[str(agent)]]
                next_solar = solar_learner.predict_solar(1)[0]
                next_consumption = next_load - next_solar

                load_learners.append(load_learner)
                solar_learners.append(solar_learner)
                next_district_consumption += next_consumption
        else:
            next_district_consumption = self.stored_district_consumptions[0]

        if next_district_consumption == 0:
            return 0
        elif next_district_consumption > 0:
            consumption_sign = 1
        else:
            consumption_sign = -1

        district_capacity = remaining_battery_capacity * num_buildings
        district_soc = soc * num_buildings

        chunk_charge_loads, stored_district_consumptions = calculate_next_chunk(observation, consumption_sign, agent_id,
                                                                                timestep,
                                                                                district_capacity, district_soc,
                                                                                load_learners, solar_learners,
                                                                                self.stored_district_consumptions)

        if agent_id == 0:
            self.stored_district_consumptions = stored_district_consumptions

        district_charge_load = -consumption_sign * chunk_charge_loads[0]
        action = district_charge_load / district_capacity

        if write_to_file:
            write_step_to_file(agent_id, timestep, action, observation)

        action += self.apply_hour_nudges(observation[2])

        return action

    def evaluate_learners(self, timestep, actual_load, actual_solar, agent_id):
        if timestep == self.evaluation_left_bound:
            # print("before", self.load_learner_best_ind)
            for forecaster in self.load_learner_options[str(agent_id)]:
                forecaster.fit_load()

            for forecaster in self.solar_learner_options[str(agent_id)]:
                forecaster.fit_solar()

        for forecaster in self.load_learner_options[str(agent_id)]:
            # forecaster.fit_load()
            prediction = forecaster.predict_load(1)[0]
            forecaster.update_values(prediction, actual_load)

        for forecaster in self.solar_learner_options[str(agent_id)]:
            # forecaster.fit_load()
            prediction = forecaster.predict_solar(1)[0]
            forecaster.update_values(prediction, actual_solar)

        if timestep == self.evaluation_right_bound:
            best_load_error = 100
            best_load_ind = 0
            for ind, forecaster in enumerate(self.load_learner_options[str(agent_id)]):
                print(ind, forecaster)
                error = forecaster.calculate_error()
                print("forecaster_error", error)
                print("best_load_error", best_load_error)
                if error < best_load_error:
                    best_load_ind = ind
                    best_load_error = error
            self.load_learner_best_ind[str(agent_id)] = best_load_ind

            best_solar_error = 100
            best_solar_ind = 0
            for ind, forecaster in enumerate(self.solar_learner_options[str(agent_id)]):
                error = forecaster.calculate_error()
                if error < best_solar_error:
                    best_solar_ind = ind
                    best_solar_error = error
            self.solar_learner_best_ind[str(agent_id)] = best_solar_ind
            print("after", self.load_learner_best_ind)
            print("after", self.solar_learner_best_ind)

    def update_load_forecasters(self, agent_id, timestep, load):
        for ind, forecaster in enumerate(self.load_learner_options[str(agent_id)]):
            model_is_used = False
            if self.evaluation_left_bound <= timestep <= self.evaluation_right_bound or self.load_learner_best_ind[
                str(agent_id)] == ind:
                model_is_used = True
            forecaster.update_loads(load, model_is_used)

    def update_solar_forecasters(self, agent_id, timestep, solar):
        for ind, forecaster in enumerate(self.solar_learner_options[str(agent_id)]):
            model_is_used = False
            if self.evaluation_left_bound <= timestep <= self.evaluation_right_bound or self.solar_learner_best_ind[
                str(agent_id)] == ind:
                model_is_used = True
            forecaster.update_solars(solar, model_is_used)

    def set_evaluation_bounds(self, timestep):
        if timestep == self.evaluation_timesteps[self.evaluation_ind]:
            self.evaluation_left_bound = self.evaluation_timesteps[self.evaluation_ind]
            self.evaluation_right_bound = self.evaluation_left_bound + self.evaluation_period
            self.evaluation_ind += 1

    def apply_hour_nudges(self, hour):
        # The best hours and nudges as calculated in 'exhaustive_hour_nudges' notebook
        if hour == 18:  # or hour == 17 or hour == 18 or hour == 20:
            return -0.02
        return 0
