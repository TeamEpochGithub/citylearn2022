import sys

from gym.spaces import Box
from agents.spinning_agent import SpinningAgent
from rewards.user_reward import UserReward

def dict_to_action_space(aspace_dict):
    return Box(
                low = aspace_dict["low"],
                high = aspace_dict["high"],
                dtype = aspace_dict["dtype"],
              )


class OrderEnforcingSpinningUpAgent:
    """
    Emulates order enforcing wrapper in Pettingzoo for easy integration
    Calls each agent step with agent in a loop and returns the action
    """
    def __init__(self, environment_arguments, model_type, model_seed, model_iteration):
        self.num_buildings = None
        self.agent = SpinningAgent(environment_arguments, model_type, model_seed, model_iteration)
        self.action_space = None
    
    def register_reset(self, observation):
        """Get the first observation after env.reset, return action""" 
        action_space = observation["action_space"]
        self.action_space = [dict_to_action_space(asd) for asd in action_space]
        obs = observation["observation"]
        self.num_buildings = len(obs)

        self.agent.set_action_space(action_space)

        # for agent_id in range(self.num_buildings):
        #     action_space = self.action_space[agent_id]
        #     self.agent.set_action_space(agent_id, action_space)
        #     self.agent.set_q_tables(agent_id)


        return self.compute_action(obs)
    
    def raise_aicrowd_error(self, msg):
        raise NameError(msg)

    def compute_action(self, observation):
        """
        Inputs: 
            observation - List of observations from the env
        Returns:
            actions - List of actions in the same order as the observations

        You can change this function as needed
        please make sure the actions are in same order as the observations

        Reward preprocesing - You can use your custom reward function here
        please specify your reward function in agents/user_agent.py

        """
        assert self.num_buildings is not None
        rewards = UserReward(agent_count=len(observation),observation=observation).calculate()

        actions = []

        return self.agent.compute_action(observation, self.num_buildings)
