"""
Main file for creating a RL agent. 

Creation date: 24/02/2024
Last modification: 28/02/024
By: Mehdi EL KANSOULI 
"""
import torch 
import torch.nn.functional as F

from agent_networks import ValueNetwork, PolicyNetwork


class BasicAgent(object):

    def __init__(self, suppliers):
        """
        Initialization of Basic Agent. 

        :params env: gym env 
            Environment modelizing the pb
        """
        self.suppliers = suppliers 
    
    def get_action(self, observation):
        """
        Policy of the basic agent. The agent return the difference between the
        forecast for the next day and the current stock. 

        :params state: env.state object

        :return dict 
            Dictionary of actions. 
        """
        actions = {
            key: max(observation.get(key)[1] - observation.get(key)[0], 0)
            for key in self.suppliers.keys()
        }

        return actions 
    

class PPOAgent(object):

    def __init__(self, obs_dim, action_dim, gamma=0.99, lr_policy=1e-3, 
                 lr_value=1e-3, epsilon=0.2):
        """
        Initialization/deifnition of the ppo agent. 

        :params obs_dim: int
            Dimension of the observation space.
        :params action_dim: int 
            Dimension of the action space. 
        :params gamma: float, default=0.99
            Discounted factor
        :params lr_policy: float, default=1e-3
            Learning rate for training policy network 
        :params lr_value: float, default=1e-3
            Learning rate for training value network 
        :params epsilon: float, default=0.2
            Clipping factor (ppo algo parameter) 

        """
        # dimension of the pb 
        self.obs_dim = obs_dim
        self.aciton_dim = action_dim

        # define vlaue and policy networks
        self.value_function = ValueNetwork(obs_dim)
        self.policy_function = PolicyNetwork(obs_dim, action_dim)

        # define parameters for training the agent. 
        self.gamma = gamma  # discounting factor
        self.epsilon = epsilon  # clipping factor
        self.lr_policy = lr_policy  
        self.lr_value = lr_value

        # define optimizers
        self.policy_opt = torch.optim.Adam(self.policy_function.parameters(), 
                                            lr=self.lr_policy)
        self.value_opt = torch.optim.Adam(self.value_function.parameters(), 
                                            lr=self.lr_value)

    def get_action(self, obs):
        """
        Given a state, sample an action from the current policy. 
        Must also provide the right format i.e. transform output tensor from 
        the policy network to a dictionary. 

        :params obs: env.state object
            Current observation of the agent

        :return dict
            Dictionary of actions
        """
        # tf ibs to a tensor 
        obs_tensor = self._handle_obs_dict(obs)

        # sample an action.
        mean, log_std = self.policy_function(obs_tensor)
        action_ = self.policy_function.sample(mean, log_std)

        # tf action into a usable dict for the env
        action = self._action_to_dict(action_)

        return action 

    def _handle_obs_dict(self, obs):
        """
        Function to transform obs given as a dictionaries to tensor usable by
        the neural networks.

        :params obs: dict
            Dict from env descrbing current obs state. 

        :return tensor
        """
        pass

    def _handle_action_dict(self, action):
        """
        Function to transform obs given as a dictionaries to tensor usable by
        the neural networks.

        :params action: dict
            Dict from env descrbing action. 

        :return tensor
        """
        pass

    def _action_to_dict(self, action):
        """
        Function that transforms the action output from the net given as a 
        tensor to a dict usable for the env. 

        :params action: tensor
            Tensor, output from policy network sample. 
        
        :return dict.
        """
        pass

    def _cumulative_reward(self, rewards):
        """
        Function used to compute the cumulative rewards given a list of rewards
        corresponding to one episode. This function returns a list as it 
        compute the disocunted reward at each step of the episode. 

        :params rewards: list

        :return list
            Discounted sum of rewards. 
        """
        # initialization 
        n = len(rewards)
        disc_rewards = [0] * n
        G = 0

        # compute discounted reward. 
        for i, reward in enumerate(rewards[::-1]):
            G *= self.gamma
            G += reward
            disc_rewards[n-i-1] = G

        G = torch.FloatTensor(G)
        return G



    def train(self, obs, actions, rewards, old_probs):
        """
        After collecting data (from an episode for example), this function 
        aims to update.

        :params obs: list of dict 
            List of all observation states stored. 
        :params acitons: list of dict
            List of all actions taken. 
        :params rewards: list
            List of reward obtained at each step. 
        :params old_probs: list
            List of old probs corresponding to actions taken 

        :return nothing
        """
        # convert all to tensor and get cumulative reward.
        obs_ = self._handle_obs_dict(obs)
        actions_ = self._handle_action_dict(actions)
        old_probs = torch.TensorFloat(old_probs)
        disc_rewards = self._cumulative_reward(rewards)

        # compute advatange /!\ must think about the estimator of adv
        adv = disc_rewards - self.value_function(obs_).squeeze()

        # compute ppo loss 
        mean, log_std = self.policy_function(obs_)
        new_probs = self.policy_function.log_prob(mean, log_std, actions_)
        ratio = torch.exp(new_probs - old_probs)
        pivot1 = ratio * adv
        pivot2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
        ppo_loss = -torch.min(pivot1, pivot2).mean()

        # compute value loss
        value_loss = F.mse_loss(self.value_function(obs_).squeeze(), 
                                disc_rewards)

        # train nn
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        ppo_loss.backward()
        value_loss.backward()
        self.policy_opt.step()
        self.value_opt.step()

        


