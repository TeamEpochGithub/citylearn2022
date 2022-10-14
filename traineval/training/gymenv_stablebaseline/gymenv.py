import gym
import numpy as np

from dataloader import DataLoader


class CityLearnEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CityLearnEnv, self).__init__()
        self.hour = -1.0
        dl = DataLoader()
        self.costs = dl.get_cost_list()
        # self.usage = dl.get_usage_list(0)
        self.battery = 0.0

        self.action_space = gym.spaces.Box(np.array([-1]), np.array([+1]), shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.array([-1.0, 0.0, 0.0]), np.array([8760.0, 1.0, 1.0]),
                                                shape=(3,), dtype=np.float32)

    def render(self, mode="human"):
        return None

    def step(self, action: np.ndarray):
        self.hour += 1.0
        observation = np.array([self.hour, self.costs[int(self.hour)], self.battery])

        # Constraint 1: Discharge can not be higher than current charge
        if -action[0] > self.battery:
            temp_action = -max(self.battery, -action[0])
            print(temp_action)
        # Constraint 2: You can not charge the battery more than full capacity minus current capacity
        elif action[0] > (1.0 - self.battery):
            temp_action = 1.0 - self.battery

        # Action matches constraints
        else:
            temp_action = action[0]

        # Finally, reward is action times current costs
        reward = -(self.costs[int(self.hour)] + (temp_action * self.costs[int(self.hour)]))
        self.battery += temp_action

        # input(f"Action: {temp_action}, cost: {self.costs[int(self.hour)]}, reward: {reward}, battery: {self.battery}")

        done = int(self.hour) == (365 * 24) - 2
        info = {}

        return np.array(observation, dtype=np.float32), float(reward), done, info

    def reset(self):
        self.hour = -1.0
        return np.array([self.hour, 0.0, 0.0], dtype=np.float32)

    def close(self):
        pass
