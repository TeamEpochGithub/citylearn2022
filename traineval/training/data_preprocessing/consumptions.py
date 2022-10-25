import pandas as pd
import numpy as np
import data.citylearn_challenge_2022_phase_1 as competition_data
import os.path as osp

b1, b2, b3, b4, b5 = (pd.read_csv(osp.join(osp.dirname(competition_data.__file__), f"Building_{i}.csv"))[
                          ["Equipment Electric Power [kWh]", "Solar Generation [W/kW]"]] for i in range(1, 6))

buildings = [b1, b2, b3, b4, b5]
solar_power = [4, 4, 4, 5, 4]
consumptions = {}

for i, b in enumerate(buildings):
    solar = b["Solar Generation [W/kW]"].to_numpy() * (solar_power[i] / 1000)
    consumption = b["Equipment Electric Power [kWh]"].to_numpy() - solar
    # consumptions.append(consumption)
    consumption = list(consumption)
    consumption.append(0)
    consumptions[f"{i}"] = consumption

df = pd.DataFrame(consumptions)
df.to_csv(osp.join(osp.dirname(competition_data.__file__), "consumptions/building_consumptions.csv"))

# final_consumption = 0
#
# for k in consumptions:
#     final_consumption += k
#
# final_consumption = list(final_consumption)
#
# array1 = pd.DataFrame(final_consumption)
# array1.to_csv('consumptions.csv')


# e = pd.read_csv("../../../analysis_data/citylearn_challenge_2022_phase_1/carbon_intensity.csv")


# for i, b in enumerate(buildings):
#     solar = b["Solar Generation [W/kW]"].to_numpy()*(solar_power[i]/1000)
#     consumption = b["Equipment Electric Power [kWh]"].to_numpy() - solar
#     #consumptions.append(consumption)
#     consumption = list(consumption)
#     consumption.append(0)
#     consumptions[f"{i}"] = consumption
#
#
# df = pd.DataFrame(consumptions)
# df.to_csv("s_consumptions.csv")
