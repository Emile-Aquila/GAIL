import torch
import torch.nn as nn
from algo import reparameterize, calc_log_pi, atanh


class ActorNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        num = 256
        self.net = nn.Sequential(
            nn.Linear(state_shape[0], num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 2 * action_shape[0]),
        )
        # state, action -> mean, log_std

    def forward(self, inputs):
        # calc means and log_stds
        means, log_stds = self.net(inputs).chunk(2, dim=-1)
        return means, log_stds

    def sample(self, inputs, deterministic=False):
        #  select action from inputs
        means, log_stds = self.forward(inputs)
        if deterministic:
            return torch.tanh(means)
        else:
            log_stds = torch.clip(log_stds, -20.0, 2.0)
            return reparameterize(means, log_stds)

    def evaluate_log_pi(self, states, actions):
        # mean，log_stdでパラメータ化した方策における，actionの確率密度の対数を算出
        means, log_stds = self.forward(states)
        noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
        return calc_log_pi(log_stds.exp(), noises, actions)


class CriticNetwork(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        num = 256
        self.net1 = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 1),
        )
        self.net2 = nn.Sequential(
            nn.Linear(state_shape[0] + action_shape[0], num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 1),
        )

    def forward(self, states, actions):
        inputs = torch.cat((states, actions), dim=-1)
        # print("inputs shape {}".format(inputs.shape))
        return self.net1(inputs), self.net2(inputs)


class CriticNetwork2(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        num = 256
        self.net1 = nn.Sequential(
            nn.Linear(state_shape[0], num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, num),
            nn.ReLU(inplace=True),
            nn.Linear(num, 1),
        )

    def forward(self, states):
        # inputs = torch.cat((states, actions), dim=-1)
        # print("inputs shape {}".format(inputs.shape))
        inputs = torch.tensor(states)
        return self.net1(inputs)
