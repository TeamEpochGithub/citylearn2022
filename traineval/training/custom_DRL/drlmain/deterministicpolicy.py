import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


class DeterministicPolicy(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_shape) -> None:
        """
            Policy network. Gives probabilities of picking actions.
        """
        super(DeterministicPolicy, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_shape[0]),
            nn.Tanh()
        )

        self.layers = []
        for i, n in enumerate(hidden_shape[:-1]):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(n, hidden_shape[i + 1]),
                    nn.Tanh()
                )
            )

        self.output = torch.nn.Linear(hidden_shape[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.input(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output(x)

        return x

    def pick(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        actions = self.forward(state)
        # actions = (((actions - (-1)) * (0.3 - (-0.3))) / (1 - (-1))) + (-0.3)
        return actions
        # return [action.item() for action in actions]
        # distribution = Categorical(probs)
        # action = distribution.sample()
        #
        # return action.item(), distribution.log_prob(action)


class QCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_shape) -> None:
        """DQN Network
        Args:
            input_dim (int): `state` dimension.
                `state` is 2-D tensor of shape (n, input_dim)
            output_dim (int): Number of actions.
                Q_value is 2-D tensor of shape (n, output_dim)
            hidden_dim (int): Hidden dimension in fc layer
        """
        super(QCritic, self).__init__()

        self.input = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_shape[0]),
            nn.Tanh()
        )

        self.layers = []
        for i, n in enumerate(hidden_shape[:-1]):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(n, hidden_shape[i + 1]),
                    nn.Tanh()
                )
            )

        self.output = torch.nn.Linear(hidden_shape[-1], 1)

    def forward(self, o: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Returns a Q_value
        Args:
            x (torch.Tensor): `State` 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: Q_value, 2-D tensor of shape (n, output_dim)
        """

        x = self.input(torch.cat((o, a), dim=1))

        for layer in self.layers:
            x = layer(x)

        x = self.output(x)

        return x


class DeterministicActorCritic:
    def __init__(self, state_shape, n_possible_outputs, estimation_depth, gamma=0.95, gradient_method='nstep',
                 learning_rate=0.01,
                 hidden_shape_actor=[16, 16], hidden_shape_critic=[16, 16], epsilon=0.01):
        self.state_shape = state_shape
        self.n_possible_outputs = n_possible_outputs
        self.n = int(estimation_depth)
        self.method = gradient_method
        self.learning_rate = learning_rate
        self.actor = DeterministicPolicy(state_shape, n_possible_outputs, hidden_shape_actor)
        self.critic = QCritic(state_shape, n_possible_outputs, hidden_shape_critic)
        self.optim_actor = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.optim_critic = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon

    def retrieve_actor(self):
        return self.actor

    def pick_greedy(self, state):
        return self.actor.pick(state)

    def pick(self, state):
        p = np.random.random()
        if p < self.epsilon:
            action = 2 * (torch.rand(1, self.n_possible_outputs) - 0.5)
        else:
            action = self.pick_greedy(state)

        action = (((action - (-1)) * (0.3 - (-0.3))) / (1 - (-1))) + (-0.3)

        return action

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.critic(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.critic(o2, self.actor(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing DDPG pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.critic(o, self.actor(o))
        return -q_pi.mean()

    def update(self, data):
        # First run one gradient descent step for Q.
        self.optim_critic.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.optim_critic.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.critic.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.optim_actor.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.optim_actor.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.critic.parameters():
            p.requires_grad = True
