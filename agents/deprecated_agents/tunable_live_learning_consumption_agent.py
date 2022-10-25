import numpy as np
from agents.helper_classes.live_learning import LiveLearner
from agents.helper_classes.tunable_live_learning import TunableLiveLearner
from traineval.utils.convert_arguments import environment_convert_argument

import os.path as osp
import joblib
import traineval.training.tpot_actions as tpot_files


def live_learning_policy(observation, action_space, live_learner, timestep, agent_id, update_learner_interval):
    live_learner.update_lists(observation)

    if timestep < 72:
        hour = observation[2]
        action = -0.067
        if 6 <= hour <= 14:
            action = 0.11
        return np.array([action], dtype=action_space.dtype)

    if timestep % update_learner_interval == 0:
        predicted_consumptions = live_learner.predict_multiple_consumptions(1, fit=True)
    else:
        predicted_consumptions = live_learner.predict_multiple_consumptions(1, fit=False)

    action = -predicted_consumptions[0] / 6.4

    action = np.array([action], dtype=action_space.dtype)
    return action


class TunableLiveLearningConsumptionAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self, params):
        self.action_space = {}
        self.cap_learning_data = params["cap_learning_data"]
        self.live_learners = {}
        self.timestep = -1
        self.update_learner_interval = params["update_learner_interval"]
        self.params = params

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        if str(agent_id) not in self.live_learners:
            self.live_learners[str(agent_id)] = TunableLiveLearner(self.cap_learning_data, self.params)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        self.timestep += 1
        return live_learning_policy(observation, self.action_space[agent_id],
                                    self.live_learners[str(agent_id)],
                                    self.timestep // len(self.live_learners), agent_id,
                                    self.update_learner_interval)
