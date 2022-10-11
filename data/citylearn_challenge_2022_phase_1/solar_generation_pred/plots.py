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
diff_consumptions = []
solar = []
loadsums = []
loads = []
prices = []
emissions = []

r = 24*3

for k in range(r):
    observations, _, done, _ = env.step(actions)
    summing = sum([observations[i][23] for i in range(5)])
    solar_sum = sum([observations[i][21] for i in range(5)])
    load_sum = sum([observations[i][20] for i in range(5)])
    price = observations[0][24]
    emissions.append(observations[0][19])

    hourly_loads = []
    hourly_consumptions = []
    for building in range(5):
        hourly_loads.append(observations[building][20])
        hourly_consumptions.append(observations[building][23])



    loads.append(hourly_loads)
    diff_consumptions.append(hourly_consumptions)

    consumptions.append(summing)
    solar.append(solar_sum)
    loadsums.append(load_sum)
    prices.append(price)

scaled_consumptions = [i/24 for i in consumptions] #scaling

clipped_consumptions = list(np.array(consumptions).clip(min=0))


def daily_average(array):
    [sum(array[h+24*k]) for h in range(24)]


# load_plot = [[loads[i][k] for i in range(r)] for k in range(5)]
# print(load_plot)
#
diff_consumptions_plot = [[diff_consumptions[i][k] for i in range(r)] for k in range(5)]

day = 0
negative_yearly_loads = []
for day in range(int(r/24)):
    building_consumptions = [0, 0, 0, 0, 0]
    for building in range(5):
        for hour in range(day*24, 24+day*24):
            c = diff_consumptions_plot[building][hour]
            if c < 0:
                building_consumptions[building] -= c
    negative_yearly_loads.append(building_consumptions)

print(negative_yearly_loads)


# plt.plot(range(r), emissions)
# plt.plot(range(r), prices)



for b in range(5):
    plt.plot(range(r), [diff_consumptions_plot[b][i]*prices[i] for i in range(r)])



# for b in range(5):
#     plt.plot(range(r), load_plot[b])

# with open("electricity_consumption", "w") as f:
#     write = csv.writer(f)
#
#     write.writerow("Load - Solar")
#     write.writerows([consumptions])

# plt.plot(range(r), scaled_consumptions, color= "blue")
# plt.scatter(range(5,r,24), [0 for i in range(5,r,24)], color="red") #Marking day at 6am
# plt.plot(range(r), [prices[i]*consumptions[i] for i in range(r)], color = "green")
# plt.plot(range(r), [emissions[i]*consumptions[i] for i in range(r)], color = "black")

# plt.plot(range(r), clipped_consumptions, color="red")
# plt.plot(range(r), prices, color="green")
# plt.plot(range(r), emissions, color="black")

# plt.plot(range(r), solar, color = "red")
# plt.plot(range(r), loadsums, color = "green")







plt.show()



# summing  = sum([observations[i][23] for i in range(5)])
# print([observations[i][23] for i in range(5)])
# print(summing)