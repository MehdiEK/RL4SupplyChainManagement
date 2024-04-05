"""
File containing neurals networks used by the PPO agent in 
order to estimate value function and policy

Creation date: 27/02/2024
Last modif: 15/03/2024
By: Mehdi 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, max_mean, max_std=10.):
        """
        Initialization of policy nektwork
        
        :params input_size: int
            Size of input, basically, observation space dim 
        :params output size: int 
            Size of output, basically, action space dim 
        :params max_mean: float 
            Maximum of action the agent can take
        :params max_std: float, default=10.
            Maximum standard deviation the agent predict
        """
        super(PolicyNetwork, self).__init__()

        # max value for bounded action space
        self.max_mean = max_mean
        self.max_std = max_std

        # layers for neural net
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.mean = nn.Linear(32, output_size)
        self.std = nn.Linear(32, output_size)

    def forward(self, obs):
        """
        Forward function 

        :params obs: array
            Observation vector
        
        :return mean and standard deviation of a normal distribution 
            for more exploration 
        """
        output = F.sigmoid(self.layer1(obs))
        output = F.sigmoid(self.layer2(output))

        # computes mean
        mean = F.sigmoid(self.mean(output))

        # compute std
        std = F.sigmoid(self.std(output))

        # scale mean and std
        mean = torch.multiply(self.max_mean, mean)
        std = torch.multiply(self.max_std, std)

        return mean, std

    def sample(self, mean, std):
        """
        Sample an action from current policy. 

        :params mean: torch.tensor
            Mean tensor of normal distribution 
        :params std: torch.tensor 
            Std of normal distribution

        :return torch.tensor
        """
        z = torch.randn_like(std)
        if np.random.random() < 0.01:
            print("\nStandard deviation: ", std)
            print("Mean: ", mean, "\n")
        action = torch.clamp(mean + std * z, 0, self.max_mean)
        return action

    def log_prob(self, mean, std, action):
        """
        Compute log proba of taking an action given a policy.

        :params mean: torch.tensor
            Mean output from the policy network
        :params std: torch.tensor
            Standard deviation of the policy nektwork 
        :params action: torch.tensor 
            Compute log proba of taking action

        :return torch.float 
            Return log probability 
        """
        z = (action - mean) / std
        log_prob = -0.5 * z**2 - std - 0.5 * np.log(2 * np.pi)
        return torch.sum(log_prob, dim=-1)
    
    def inference(self, obs):
        """
        Inference function 

        :params obs: array
            Observation vector
        :params torch.tensor 
            Return mean without any exploration.
        """
        mean, _ = self.forward(obs)
        return mean
    

class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        """
        Initialization of value network. 

        :params input_size: int 
            Input shape, basically dimension of observation space.
        """
        super(ValueNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 32)
        self.layer2 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        value = self.output(x)
        return value