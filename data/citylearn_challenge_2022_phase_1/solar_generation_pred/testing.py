from citylearn.citylearn import CityLearnEnv
import numpy as np
from gym.spaces import Box
import matplotlib.pyplot as plt
import csv



class Constants:
    episodes = 5
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'


env = CityLearnEnv(schema="C:/Users/bjorn/OneDrive/Documents/TU Delft/EPOCH/CityLearn/citylearn-2022-starter-kit/data/citylearn_challenge_2022_phase_1/schema.json")

action = 0
action_space = Box(-1.0, 1.0, (1,), np.float32)


action = np.array([action], dtype=action_space.dtype)

actions = [action for _ in range(5)]

consumptions = []
solar = []
loadsums = []
loads = []
prices = []

r = 24*7

for k in range(r):
    observations, _, done, _ = env.step(actions)
    summing = sum([observations[i][23] for i in range(5)])
    solar_sum = sum([observations[i][21] for i in range(5)])
    load_sum = sum([observations[i][20] for i in range(5)])
    price = observations[0][24]

    hourly_loads = []
    for building in range(5):
        hourly_loads.append(observations[building][20])


    loads.append(hourly_loads)

    consumptions.append(summing)
    solar.append(solar_sum)
    loadsums.append(load_sum)
    prices.append(price)

# consumptions = [i/4 for i in consumptions] #scaling

clipped_consumptions = list(np.array(consumptions).clip(min=0))

def daily_average(array):
    [sum(array[h+24*k]) for h in range(24)]


# load_plot = [[loads[i][k] for i in range(r)] for k in range(5)]
# print(load_plot)
#
# for b in range(5):
#     plt.plot(range(r), load_plot[b])

# with open("electricity_consumption", "w") as f:
#     write = csv.writer(f)
#
#     write.writerow("Load - Solar")
#     write.writerows([consumptions])

plt.plot(range(r), consumptions, color= "blue")
# plt.plot(range(r), clipped_consumptions, color="red")
# plt.plot(range(r), prices, color="green")

plt.plot(range(r), solar, color = "red")
plt.plot(range(r), loadsums, color = "green")




plt.show()




# summing  = sum([observations[i][23] for i in range(5)])
# print([observations[i][23] for i in range(5)])
# print(summing)