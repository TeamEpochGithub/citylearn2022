import numpy as np
import os.path as osp
import csv

from agents.helper_classes.live_learning_load_error import LiveLearnerLoadError
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


def get_chunk_consumptions_fit_delay(consumption_sign, live_learner, timestep):
    max_chunk_size = 32

    if timestep + max_chunk_size > 8759:
        max_chunk_size = 8759 - timestep

    chunk_consumptions = live_learner.predict_consumption(max_chunk_size, False)

    for index, consumption in enumerate(chunk_consumptions):

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
                         live_learner):
    chunk_consumptions = get_chunk_consumptions_fit_delay(consumption_sign, live_learner, timestep)

    if consumption_sign == -1:  # If negative consumption
        chunk_charge_loads = negative_consumption_scenario(chunk_consumptions, remaining_battery_capacity, soc)
    else:
        date = shift_date(observation[2], observation[1], observation[0], shifts=1)
        chunk_charge_loads = positive_consumption_scenario(date, chunk_consumptions, soc)

    return chunk_charge_loads


def pred_consumption_policy(observation, timestep, agent_id, remaining_battery_capacity, soc, live_learner,
                            write_to_file, params):
    if timestep >= 8758:
        return -1

    live_learner.update_lists(observation)
    print("timestep: ", timestep)

    hour = observation[2]
    print(params)
    if hour == params["hour_0"]:
        return params["action_0"]

    if hour == params["hour_1"]:
        return params["action_1"]

    if hour == params["hour_2"]:
        return params["action_2"]

    if hour == params["hour_3"]:
        return params["action_3"]

    if hour == params["hour_4"]:
        return params["action_4"]

    if timestep < 72:
        action = -0.067
        if 6 <= hour <= 14:
            action = 0.11
        return action

    next_consumption = live_learner.predict_consumption(1, True)[0]

    if next_consumption == 0:
        return 0
    elif next_consumption > 0:
        consumption_sign = 1
    else:
        consumption_sign = -1

    chunk_charge_loads = calculate_next_chunk(observation, consumption_sign, agent_id, timestep,
                                              remaining_battery_capacity, soc, live_learner)

    charge_load = -1 * consumption_sign * chunk_charge_loads[0]
    action = charge_load / remaining_battery_capacity

    if write_to_file:
        write_step_to_file(agent_id, timestep, action, observation)

    return action


class TunableTimeStepPredConsumptionAgentPeak:

    def __init__(self, params):
        self.action_space = {}
        self.timestep = -1
        self.remaining_battery_capacity = {}
        self.soc = {}
        self.write_to_file = False
        self.params = params

        self.live_learners = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.remaining_battery_capacity[agent_id] = 6.4
        self.soc[agent_id] = 0

        if str(agent_id) not in self.live_learners:
            self.live_learners[str(agent_id)] = LiveLearner(800, 1, self.write_to_file, agent_id)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        self.timestep += 1
        building_timestep = self.timestep // len(observation)
        observation = observation[agent_id]

        action_out = \
            pred_consumption_policy(observation, building_timestep, agent_id,
                                    self.remaining_battery_capacity[agent_id],
                                    self.soc[agent_id], self.live_learners[str(agent_id)],
                                    self.write_to_file, self.params)

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