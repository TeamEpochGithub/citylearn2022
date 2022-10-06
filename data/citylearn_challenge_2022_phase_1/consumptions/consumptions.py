import pandas as pd
import numpy as np

b1, b2, b3, b4, b5 = (pd.read_csv(f"../Building_{i}.csv")[["Equipment Electric Power [kWh]", "Solar Generation [W/kW]"]] for i in range(1, 6))

buildings = [b1, b2, b3, b4, b5]
solar_power = [4, 4, 4, 5, 4]
consumptions = []

for i, b in enumerate(buildings):
    solar = b["Solar Generation [W/kW]"].to_numpy()*(solar_power[i]/1000)
    consumption = b["Equipment Electric Power [kWh]"].to_numpy() - solar
    consumptions.append(consumption)

final_consumption = 0

for k in consumptions:
    final_consumption += k

final_consumption = list(final_consumption)

array1 = pd.DataFrame(final_consumption)
array1.to_csv('consumptions.csv')



