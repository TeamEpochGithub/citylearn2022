import numpy as np
from agents.helper_classes.live_learning import LiveLearner
from traineval.utils.convert_arguments import environment_convert_argument

import os.path as osp
import joblib
import traineval.training.tpot_actions as tpot_files


def live_learning_and_tpot_policy(observation, action_space, live_learner, action_model, timestep, agent_id, update_learner_interval):
    if timestep > 8750:
        return np.array([-0.1], dtype=action_space.dtype)

    live_learner.update_lists(observation)

    if timestep < 72:
        hour = observation[2]
        action = -0.067
        if 6 <= hour <= 14:
            action = 0.11
        return np.array([action], dtype=action_space.dtype)

    col_names = ["month",
                 "day_type",
                 "hour",
                 "non_shiftable_load",
                 "solar_generation",
                 "electrical_storage_soc",
                 "net_electricity_consumption"]

    obs = []
    for i in environment_convert_argument(col_names):
        obs.append(observation[i])

    if timestep % update_learner_interval == 0:
        predicted_consumptions = live_learner.predict_multiple_consumptions(6, fit=True)
    else:
        predicted_consumptions = live_learner.predict_multiple_consumptions(6, fit=False)

    action = action_model.predict([obs + predicted_consumptions])
    if agent_id == 1:
        print(timestep, action)

    action = np.array([action], dtype=action_space.dtype)
    return action


class LiveLearningAgentTPOTActions:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self):
        self.action_space = {}
        self.cap_learning_data = 1000
        self.live_learners = {}
        tpot_model_path = osp.join(osp.dirname(tpot_files.__file__), 'pipe.joblib')
        self.action_model = joblib.load(tpot_model_path)
        self.timestep = -1
        self.update_learner_interval = 3

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        if str(agent_id) not in self.live_learners:
            self.live_learners[str(agent_id)] = LiveLearner(self.cap_learning_data)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        self.timestep += 1
        return live_learning_and_tpot_policy(observation, self.action_space[agent_id],
                                             self.live_learners[str(agent_id)],
                                             self.action_model, self.timestep // len(self.live_learners), agent_id, self.update_learner_interval)
