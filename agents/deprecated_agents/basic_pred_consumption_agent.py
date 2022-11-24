import numpy as np

from agents.helper_classes.live_learning import LiveLearner


def individual_consumption_policy(action_space, timestep, agent_id, observation, live_learner):
    if timestep >= 8759:
        return np.array([0], dtype=action_space.dtype)

    live_learner.update_lists(observation)

    if timestep < 60:
        hour = observation[2]
        action = -0.067
        if 6 <= hour <= 14:
            action = 0.11
        return np.array([action], dtype=action_space.dtype)

    next_consumption = live_learner.predict_consumption(1)[0]

    action = -next_consumption / 6.4

    action = np.array([action], dtype=action_space.dtype)
    return action


class BasicPredConsumptionAgent:

    def __init__(self):
        self.action_space = {}
        self.timestep = -1
        self.live_learners = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        if str(agent_id) not in self.live_learners:
            self.live_learners[str(agent_id)] = LiveLearner(800, 30)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        self.timestep += 1
        building_timestep = self.timestep // len(observation)
        observation = observation[agent_id]

        return individual_consumption_policy(self.action_space[agent_id], building_timestep, agent_id, observation,
                                             self.live_learners[str(agent_id)])
