from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from citylearn.agents.rbc import RBC, BasicRBC, OptimizedRBC
from agents.rlc_agent import RLCAgent
from citylearn.rl import PolicyNetwork, ReplayBuffer, SoftQNetwork


class SAC(RLCAgent):
    def __init__(self, *args, **kwargs):
        r"""Initialize :class:`SAC`.
        Parameters
        ----------
        *args : tuple
            `RLC` positional arguments.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(*args, **kwargs)

        # internally defined
        self.action_space = {}
        self.__normalized = False
        self.__alpha = 0.2
        self.__soft_q_criterion = nn.SmoothL1Loss()
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__replay_buffer = ReplayBuffer(int(self.replay_buffer_capacity))
        self.__soft_q_net1 = None
        self.__soft_q_net2 = None
        self.__target_soft_q_net1 = None
        self.__target_soft_q_net2 = None
        self.__policy_net = None
        self.__soft_q_optimizer1 = None
        self.__soft_q_optimizer2 = None
        self.__policy_optimizer = None
        self.__target_entropy = None
        self.__norm_mean = None
        self.__norm_std = None
        self.__r_norm_mean = None
        self.__r_norm_std = None
        self.__set_networks()

    @property
    def device(self) -> torch.device:
        """Device; cuda or cpu."""

        return self.__device

    @property
    def soft_q_net1(self) -> SoftQNetwork:
        """soft_q_net1."""

        return self.__soft_q_net1

    @property
    def soft_q_net2(self) -> SoftQNetwork:
        """soft_q_net2."""

        return self.__soft_q_net2

    @property
    def policy_net(self) -> PolicyNetwork:
        """policy_net."""

        return self.__policy_net

    @property
    def norm_mean(self) -> List[float]:
        """norm_mean."""

        return self.__norm_mean

    @property
    def norm_std(self) -> List[float]:
        """norm_std."""

        return self.__norm_std

    @property
    def normalized(self) -> bool:
        """normalized."""

        return self.__normalized

    @property
    def r_norm_mean(self) -> float:
        """r_norm_mean."""

        return self.__r_norm_mean

    @property
    def r_norm_std(self) -> float:
        """r_norm_std."""

        return self.__r_norm_std

    @property
    def replay_buffer(self) -> ReplayBuffer:
        """replay_buffer."""

        return self.__replay_buffer

    @property
    def alpha(self) -> float:
        """alpha."""

        return self.__alpha

    @property
    def soft_q_criterion(self) -> nn.SmoothL1Loss:
        """soft_q_criterion."""

        return self.__soft_q_criterion

    @property
    def target_soft_q_net1(self) -> SoftQNetwork:
        """target_soft_q_net1."""

        return self.__target_soft_q_net1

    @property
    def target_soft_q_net2(self) -> SoftQNetwork:
        """target_soft_q_net2."""

        return self.__target_soft_q_net2

    @property
    def soft_q_optimizer1(self) -> optim.Adam:
        """soft_q_optimizer1."""

        return self.__soft_q_optimizer1

    @property
    def soft_q_optimizer2(self) -> optim.Adam:
        """soft_q_optimizer2."""

        return self.__soft_q_optimizer2

    @property
    def policy_optimizer(self) -> optim.Adam:
        """policy_optimizer."""

        return self.__policy_optimizer

    @property
    def target_entropy(self) -> float:
        """target_entropy."""

        return self.__target_entropy

    def add_to_buffer(self, observations: List[float], actions: List[float], reward: float,
                      next_observations: List[float], done: bool = False):
        r"""Update replay buffer.
        Parameters
        ----------
        observations : List[float]
            Previous time step observations.
        actions : List[float]
            Previous time step actions.
        reward : float
            Current time step reward.
        next_observations : List[float]
            Current time step observations.
        done : bool
            Indication that episode has ended.
        """

        # Run once the regression model has been fitted
        # Normalize all the observations using periodical normalization, one-hot encoding, or -1, 1 scaling. It also removes observations that are not necessary (solar irradiance if there are no solar PV panels).
        observations = np.array(self.__get_encoded_observations(observations), dtype=float)
        next_observations = np.array(self.__get_encoded_observations(next_observations), dtype=float)

        if self.normalized:
            observations = np.array(self.__get_normalized_observations(observations), dtype=float)
            next_observations = np.array(self.__get_normalized_observations(observations), dtype=float)
            reward = self.__get_normalized_reward(reward)
        else:
            pass

        self.__replay_buffer.push(observations, actions, reward, next_observations, done)

        if self.time_step >= self.start_training_time_step and self.batch_size <= len(self.__replay_buffer):
            if not self.normalized:
                X = np.array([j[0] for j in self.__replay_buffer.buffer], dtype=float)
                self.__norm_mean = np.nanmean(X, axis=0)
                self.__norm_std = np.nanstd(X, axis=0) + 1e-5
                R = np.array([j[2] for j in self.__replay_buffer.buffer], dtype=float)
                self.__r_norm_mean = np.nanmean(R, dtype=float)
                self.__r_norm_std = np.nanstd(R, dtype=float) / self.reward_scaling + 1e-5
                new_buffer = [(
                    np.hstack(
                        (np.array(self.__get_normalized_observations(observations), dtype=float)).reshape(1, -1)[0]),
                    actions,
                    self.__get_normalized_reward(reward),
                    np.hstack(
                        (np.array(self.__get_normalized_observations(next_observations), dtype=float)).reshape(1, -1)[
                            0]),
                    done
                ) for observations, actions, reward, next_observations, done in self.__replay_buffer.buffer]
                self.__replay_buffer.buffer = new_buffer
                self.__normalized = True
            else:
                pass

            for _ in range(self.update_per_time_step):
                observations, actions, reward, next_observations, done = self.__replay_buffer.sample(self.batch_size)
                tensor = torch.cuda.FloatTensor if self.device.type == 'cuda' else torch.FloatTensor
                observations = tensor(observations).to(self.device)
                next_observations = tensor(next_observations).to(self.device)
                actions = tensor(actions).to(self.device)
                reward = tensor(reward).unsqueeze(1).to(self.device)
                done = tensor(done).unsqueeze(1).to(self.device)

                with torch.no_grad():
                    # Update Q-values. First, sample an action from the Gaussian policy/distribution for the current (next) observation and its associated log probability of occurrence.
                    new_next_actions, new_log_pi, _ = self.__policy_net.sample(next_observations)

                    # The updated Q-value is found by subtracting the logprob of the sampled action (proportional to the entropy) to the Q-values estimated by the target networks.
                    target_q_values = torch.min(
                        self.__target_soft_q_net1(next_observations, new_next_actions),
                        self.__target_soft_q_net2(next_observations, new_next_actions),
                    ) - self.alpha * new_log_pi
                    q_target = reward + (1 - done) * self.discount * target_q_values

                # Update Soft Q-Networks
                q1_pred = self.__soft_q_net1(observations, actions)
                q2_pred = self.__soft_q_net2(observations, actions)
                q1_loss = self.__soft_q_criterion(q1_pred, q_target)
                q2_loss = self.__soft_q_criterion(q2_pred, q_target)
                self.__soft_q_optimizer1.zero_grad()
                q1_loss.backward()
                self.__soft_q_optimizer1.step()
                self.__soft_q_optimizer2.zero_grad()
                q2_loss.backward()
                self.__soft_q_optimizer2.step()

                # Update Policy
                new_actions, log_pi, _ = self.__policy_net.sample(observations)
                q_new_actions = torch.min(
                    self.__soft_q_net1(observations, new_actions),
                    self.__soft_q_net2(observations, new_actions)
                )
                policy_loss = (self.alpha * log_pi - q_new_actions).mean()
                self.__policy_optimizer.zero_grad()
                policy_loss.backward()
                self.__policy_optimizer.step()

                # Soft Updates
                for target_param, param in zip(self.__target_soft_q_net1.parameters(), self.__soft_q_net1.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

                for target_param, param in zip(self.__target_soft_q_net2.parameters(), self.__soft_q_net2.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        else:
            pass

    def select_actions(self, observation, action_space):
        r"""Provide actions for current time step.
        Will return randomly sampled actions from `action_space` if :attr:`end_exploration_time_step` >= :attr:`time_step`
        else will use policy to sample actions.

        Returns
        -------
        actions: List[float]
            Action values
        """

        if self.time_step <= self.end_exploration_time_step:
            action = self.get_exploration_actions(observation)

        else:
            action = self.__get_post_exploration_actions(observation)

        action = np.array([action], dtype=action_space.dtype)
        assert action_space.contains(action)
        return action

    def __get_post_exploration_actions(self, observation):
        """Action sampling using policy, post-exploration time step"""

        observations = (self.__get_encoded_observations(observation))
        observations = (self.__get_normalized_observations(observations))
        observations = torch.FloatTensor(observations).unsqueeze(0).to(self.__device)
        result = self.__policy_net.sample(observations)
        actions = result[2] if self.time_step >= self.deterministic_start_time_step else result[0]
        actions = actions.detach().cpu().numpy()[0]
        return actions

    def get_exploration_actions(self, observations):
        """Return randomly sampled actions from `action_space` multiplied by :attr:`action_scaling_coefficient`.

        Returns
        -------
        actions: List[float]
            Action values.
        """

        # random actions
        return self.action_scaling_coefficient * self.action_space.sample()

    def __get_normalized_reward(self, reward: float) -> float:
        return (reward - self.r_norm_mean) / self.r_norm_std

    def __get_normalized_observations(self, observations):
        return ((np.array(observations, dtype=float) - self.norm_mean) / self.norm_std).tolist()

    def __get_encoded_observations(self, observation):
        return np.array([j for j in np.hstack(self.encoders * np.array(observation, dtype=float)) if j != None],
                        dtype=float).tolist()

    def __set_networks(self):
        # init networks
        self.__soft_q_net1 = SoftQNetwork(self.observation_dimension, self.action_dimension, self.hidden_dimension).to(
            self.device)
        self.__soft_q_net2 = SoftQNetwork(self.observation_dimension, self.action_dimension, self.hidden_dimension).to(
            self.device)
        self.__target_soft_q_net1 = SoftQNetwork(self.observation_dimension, self.action_dimension,
                                                 self.hidden_dimension).to(self.device)
        self.__target_soft_q_net2 = SoftQNetwork(self.observation_dimension, self.action_dimension,
                                                 self.hidden_dimension).to(self.device)

        for target_param, param in zip(self.__target_soft_q_net1.parameters(), self.__soft_q_net1.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.__target_soft_q_net2.parameters(), self.__soft_q_net2.parameters()):
            target_param.data.copy_(param.data)

        # Policy
        self.__policy_net = PolicyNetwork(self.observation_dimension, self.action_dimension, self.action_space,
                                          self.action_scaling_coefficient, self.hidden_dimension).to(self.device)
        self.__soft_q_optimizer1 = optim.Adam(self.__soft_q_net1.parameters(), lr=self.lr)
        self.__soft_q_optimizer2 = optim.Adam(self.__soft_q_net2.parameters(), lr=self.lr)
        self.__policy_optimizer = optim.Adam(self.__policy_net.parameters(), lr=self.lr)
        self.__target_entropy = -np.prod(self.action_dimension).item()

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return self.select_actions(observation, self.action_space[agent_id])