import numpy as np
import os.path as osp
import csv

from agents.helper_classes.load_learner_instances import Lag0Learner, Lag1Learner, Lag2Learner, Lag3Learner, \
    Lag4Learner, \
    Lag5Learner, Lag6Learner

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


def get_chunk_consumptions_fit_delay(consumption_sign, load_learner, solar_learner, timestep):
    max_chunk_size = 32

    if timestep + max_chunk_size > 8759:
        max_chunk_size = 8759 - timestep

    loads = load_learner.predict_load(max_chunk_size)
    solars = solar_learner.predict_solar_generations(max_chunk_size)
    # print("loads", loads)
    # print("solars", solars)

    chunk_consumptions = [a - b for a, b in zip(loads, solars)]
    # POS
    for index, consumption in enumerate(chunk_consumptions):
        # NEG
        if consumption * consumption_sign < 0:
            chunk_consumptions = chunk_consumptions[:index]
            break

    return chunk_consumptions


def negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc):
    chunk_total_consumption = sum(chunk_consumptions)

    if -1 * chunk_total_consumption >= (remaining_battery_capacity - soc) / np.sqrt(0.83):
        # If more energy can be obtained than the one necessary to charge the battery
        relative_consumption = [i / chunk_total_consumption for i in chunk_consumptions]
        chunk_charge_loads = [i * (remaining_battery_capacity - soc) / np.sqrt(0.83) for i in relative_consumption]
    else:  # Otherwise charge with all the possible energy
        chunk_charge_loads = [-1 * i for i in chunk_consumptions]

    return chunk_charge_loads


def positive_consumption_scenario(obs_date, chunk_consumptions, soc):
    chunk_total_consumption = sum(chunk_consumptions)

    if chunk_total_consumption >= soc * np.sqrt(0.83):
        # If fully discharging the battery doesn't bring the consumption to 0, we take the highest
        # price*consumption value and bring it down to the next highest price*consumption by reducing the
        # consumption at that time step. We do this consecutively until the battery has been emptied.

        date = obs_date
        prices = []

        for hour in range(len(chunk_consumptions)):
            prices.append(pricing(date[2], date[0], date[1]))
            date = shift_date(date[0], date[1], date[2], shifts=1)

        consumption_prices = [prices[i] * c for i, c in enumerate(chunk_consumptions)]

        local_soc = soc * np.sqrt(0.83)
        chunk_charge_loads = [0] * len(chunk_consumptions)

        return lowering_peaks(local_soc, chunk_charge_loads, consumption_prices, prices)

    else:
        return chunk_consumptions


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


def calculate_next_chunk(observation, consumption_sign, agent_id, timestep, remaining_battery_capacity, soc,
                         load_learner, solar_learner):
    chunk_consumptions = get_chunk_consumptions_fit_delay(consumption_sign, load_learner, solar_learner, timestep)
    # print("chunk_consumptions", chunk_consumptions)

    if consumption_sign == -1:  # If negative consumption
        chunk_charge_loads = negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc)
    else:
        date = shift_date(observation[2], observation[1], observation[0], shifts=1)
        chunk_charge_loads = positive_consumption_scenario(date, chunk_consumptions, soc)
    # print("chunk_charge_loads", chunk_charge_loads)
    return chunk_charge_loads


def day_night_policy(hour):
    action = -0.067
    if 6 <= hour <= 14:
        action = 0.11
    return action


class TunableTimeStepPredConsumptionAgentPeak:

    def __init__(self, params):
        self.action_space = {}
        self.timestep = -1
        self.remaining_battery_capacity = {}
        self.soc = {}
        self.write_to_file = False

        self.live_learners = {}
        self.load_learner_options = {}
        self.load_learner_best_ind = {}
        self.evaluation_timesteps = [500, 2000, 5000, -10]  # -10 added to prevent 'index out of range'
        self.evaluation_ind = 0
        self.evaluation_period = 168
        self.evaluation_left_bound = -10
        self.evaluation_right_bound = -10
        self.params = params

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.remaining_battery_capacity[agent_id] = 6.4
        self.soc[agent_id] = 0

        if str(agent_id) not in self.live_learners:
            self.live_learners[str(agent_id)] = LiveLearner(800, 15, self.write_to_file, agent_id)

        if str(agent_id) not in self.load_learner_best_ind:
            self.load_learner_best_ind[str(agent_id)] = 0

        if str(agent_id) not in self.load_learner_options:
            self.load_learner_options[str(agent_id)] = [Lag0Learner(800, 15),
                                                        Lag1Learner(800, 15),
                                                        Lag2Learner(800, 15),
                                                        Lag3Learner(800, 15),
                                                        Lag4Learner(800, 15),
                                                        Lag5Learner(800, 15),
                                                        Lag6Learner(800, 15)
                                                        ]

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        building_timestep = self.timestep // len(observation)
        observation = observation[agent_id]

        action_out = \
            self.pred_consumption_policy(observation, building_timestep, agent_id,
                                         self.remaining_battery_capacity[agent_id],
                                         self.soc[agent_id], self.live_learners[str(agent_id)],
                                         self.write_to_file)

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

    def pred_consumption_policy(self, observation, timestep, agent_id, remaining_battery_capacity, soc,
                                solar_live_learner, write_to_file):
        if timestep >= 8758:
            return -1

        self.set_evaluation_bounds(timestep)
        self.update_load_forecasters(agent_id, timestep, observation[20])
        solar_live_learner.update_lists(observation)

        print("timestep: ", timestep)

        if timestep < 72:
            return day_night_policy(observation[2])

        if self.evaluation_left_bound <= timestep <= self.evaluation_right_bound:
            self.evaluate_learners(timestep, observation[20], agent_id)

        load_learner = self.load_learner_options[str(agent_id)][self.load_learner_best_ind[str(agent_id)]]
        next_load = load_learner.predict_load(1)[0]
        next_solar = solar_live_learner.predict_solar_generations(1)[0]
        next_consumption = next_load - next_solar

        if next_consumption == 0:
            return 0
        elif next_consumption > 0:
            consumption_sign = 1
        else:
            consumption_sign = -1

        chunk_charge_loads = calculate_next_chunk(observation, consumption_sign, agent_id, timestep,
                                                  remaining_battery_capacity, soc, load_learner, solar_live_learner)

        charge_load = -consumption_sign * chunk_charge_loads[0]
        action = charge_load / remaining_battery_capacity

        if write_to_file:
            write_step_to_file(agent_id, timestep, action, observation)

        action += self.nudge_action(observation[2])

        return action

    def evaluate_learners(self, timestep, actual_load, agent_id):
        if timestep == self.evaluation_left_bound:
            print("before", self.load_learner_best_ind)
            for forecaster in self.load_learner_options[str(agent_id)]:
                forecaster.fit_load()

        for forecaster in self.load_learner_options[str(agent_id)]:
            # forecaster.fit_load()
            prediction = forecaster.predict_load(1)[0]
            forecaster.update_values(prediction, actual_load)

        if timestep == self.evaluation_right_bound:
            best_error = 100
            best_ind = 0
            for ind, forecaster in enumerate(self.load_learner_options[str(agent_id)]):
                error = forecaster.calculate_error()
                if error < best_error:
                    best_ind = ind
                    best_error = error
            self.load_learner_best_ind[str(agent_id)] = best_ind
            print("after", self.load_learner_best_ind)

    def update_load_forecasters(self, agent_id, timestep, load):
        for ind, forecaster in enumerate(self.load_learner_options[str(agent_id)]):
            model_is_used = False
            if self.evaluation_left_bound <= timestep <= self.evaluation_right_bound or self.load_learner_best_ind[
                str(agent_id)] == ind:
                model_is_used = True
            forecaster.update_loads(load, model_is_used)

    def set_evaluation_bounds(self, timestep):
        if timestep == self.evaluation_timesteps[self.evaluation_ind]:
            self.evaluation_left_bound = self.evaluation_timesteps[self.evaluation_ind]
            self.evaluation_right_bound = self.evaluation_left_bound + self.evaluation_period
            print(self.evaluation_left_bound, self.evaluation_right_bound)
            self.evaluation_ind += 1

    def nudge_action(self, hour):
        nudge = 0
        if hour == self.params["hour_0"]:
            nudge += self.params["action_0"]
        elif hour == self.params["hour_1"]:
            nudge += self.params["action_1"]
        elif hour == self.params["hour_2"]:
            nudge += self.params["action_2"]
        # elif hour == self.params["hour_3"]:
        #     nudge += self.params["action_3"]
        # elif hour == self.params["hour_4"]:
        #     nudge += self.params["action_4"]
        return nudge
