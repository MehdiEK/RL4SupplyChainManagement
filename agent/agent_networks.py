"""
File containing neurals networks used by the PPO agent in 
order to estimate value function and policy

Creation date: 27/02/2024
Last modif: 27/02/2024
By: Mehdi 
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initialization of policy nektwork
        
        :params input_size: int
            Size of input, basically, observation space dim 
        :params output size: int 
            Size of output, basically, action space dim 
        """
        super(PolicyNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.mean_layer = nn.Linear(32, output_size)
        self.log_std_layer = nn.Linear(32, output_size)

    def forward(self, obs):
        """
        Forward function 

        :params obs: array
            Observation vector
        """
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        return mean, log_std

    def sample(self, mean, log_std):
        """
        Sample an action from current policy. 
        """
        std = torch.exp(0.5 * log_std)
        z = torch.randn_like(std)
        action = mean + std * z
        return action

    def log_prob(self, mean, log_std, action):
        std = torch.exp(0.5 * log_std)
        z = (action - mean) / std
        log_prob = -0.5 * z.pow(2) - log_std - 0.5 * np.log(2 * np.pi)
        return log_prob.sum(dim=-1)