"""
Main file for creating a RL agent. 

Creation date: 24/02/2024
Last modification: 15/03/2024
By: Mehdi EL KANSOULI 
"""
import torch 
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from agent.agent_networks import ValueNetwork, PolicyNetwork


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
    
    def inference(self, observation):
        """
        """
        return self.get_action(observation)

class PPOAgent(object):

    def __init__(self, suppliers:dict, obs_dim:int,  gamma=0.95, 
                 lr_policy=1e-3, lr_value=1e-3, epsilon=0.2, 
                 max_action=250):
        """
        Initialization/deifnition of the ppo agent. 

        :params obs_dim: int
            Dimension of the observation space.
        :params action_dim: int 
            Dimension of the action space. 
        :params gamma: float, default=0.5
            Discounted factor
        :params lr_policy: float, default=1e-3
            Learning rate for training policy network 
        :params lr_value: float, default=1e-3
            Learning rate for training value network 
        :params epsilon: float, default=0.2
            Clipping factor (ppo algo parameter) 

        """
        # dimension of the pb 
        self.suppliers = list(suppliers.keys())
        self.obs_dim = obs_dim
        self.action_dim = len(suppliers.keys())

        # define vlaue and policy networks
        self.value_function = ValueNetwork(obs_dim)
        self.policy_function = PolicyNetwork(obs_dim, 
                                             self.action_dim, 
                                             max_mean=max_action)

        # define parameters for training the agent. 
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # clipping factor
        self.lr_policy = lr_policy  
        self.lr_value = lr_value

        # define optimizers
        self.policy_opt = torch.optim.Adam(self.policy_function.parameters(), 
                                           lr=self.lr_policy)
        self.value_opt = torch.optim.Adam(self.value_function.parameters(), 
                                            lr=self.lr_value)

    def _handle_obs_dict(self, obs):
        """
        Function to transform obs given as a dictionaries to tensor usable by
        the neural networks.

        :params obs: list of dict
            Dict from env descrbing current obs state. 

        :return tensor
        """
        # if single create a list of one element
        if not isinstance(obs, list):
            obs = [obs]
        
        obs_lists = []
        for o in obs: 
            pivot_list = []
            for val in o.values():
                pivot_list = pivot_list + list(val)
            obs_lists.append(pivot_list)

        obs_tensor = torch.Tensor(obs_lists)

        return obs_tensor

    def _handle_action_dict(self, actions):
        """
        Function to transform action given as a dictionaries to tensor usable by
        the neural networks.

        :params action: dict
            Dict from env descrbing action. 

        :return tensor
        """
        # if single create a list of one element
        if not isinstance(actions, list):
            actions = [actions]

        # actions_list = [list(action.values()) for action in actions]

        actions_lists = []
        for action in actions: 
            pivot_list = []
            for val in action.values():
                pivot_list.append(val)
            actions_lists.append(pivot_list)

        actions_tensor = torch.Tensor(actions_lists)

        return actions_tensor

    def _action_to_dict(self, action):
        """
        Function that transforms the action output from the net given as a 
        tensor to a dict usable for the env. 

        :params action: tensor
            Tensor, output from policy network sample. Must be exactly one 
            action 
        
        :return dict.
        """
        # get only the 1-st dim
        action = action.squeeze()

        # create a dict
        actions_dict = {}
        for i in range(self.action_dim):
            actions_dict[self.suppliers[i]] = action[i].item()
        
        action_ = actions_dict  # Dict(actions_dict)
        return action_

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
        mean, std = self.policy_function(obs_tensor)
        action_ = self.policy_function.sample(mean, std)

        # tf action into a usable dict for the env
        action = self._action_to_dict(action_)

        return action 
    
    def get_probs(self, obs, action):
        """
        Get probability of an action given that the agent is in 
        a state observed as obs.

        :params obs: gym Dict 
            Observation state 
        :params action: gym Dict 
            Action take given obs
        
            :return float
        """
        action_ = self._handle_action_dict(action)
        obs_ = self._handle_obs_dict(obs)

        mean, std = self.policy_function(obs_)
        prob = self.policy_function.log_prob(mean, std, action_)

        return prob

    def _cumulative_reward(self, rewards, last_value):
        """
        Function used to compute the cumulative rewards given a list of rewards
        corresponding to one episode. This function returns a list as it 
        compute the disocunted reward at each step of the episode. 

        :params rewards: list

        :params last_value: float
            Value function evaluation on last obs

        :return list
            Discounted sum of rewards. 
        """
        # initialization 
        n = len(rewards) 
        G = last_value
        disc_rewards = [0] * n

        # compute discounted reward. 
        for i, reward in enumerate(rewards[::-1]):
            G *= self.gamma
            G += reward
            disc_rewards[n-i-1] = G

        disc_rewards = torch.tensor(disc_rewards, dtype=torch.float32)
        return disc_rewards

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
        with torch.no_grad():
            # convert all to tensor and get cumulative reward.
            obs_ = self._handle_obs_dict(obs)
            actions_ = self._handle_action_dict(actions)
            old_probs = torch.Tensor(old_probs)

            # set last reward equal to the evaluation of V
            disc_rewards = self._cumulative_reward(
                rewards, 
                self.value_function(obs_[-1]).squeeze()                                      
            )

            # compute advatange /!\ must think about the estimator of adv
        
        adv = disc_rewards - self.value_function(obs_[:-1]).squeeze()
        # adv = (adv - torch.mean(adv)) / (torch.std(adv) + 1e-8)

        # compute ppo loss 
        mean, std = self.policy_function(obs_[:-1])
        new_probs = self.policy_function.log_prob(mean, std, actions_)
        ratio = torch.exp(new_probs - old_probs)
        pivot1 = torch.multiply(ratio, adv)
        pivot2 = torch.multiply(torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon), adv)
        ppo_loss = - torch.mean(torch.min(pivot1, pivot2))

        # compute value loss
        value_loss = F.mse_loss(self.value_function(obs_[:-1]).squeeze(), 
                                disc_rewards)

        # print("\nPPO loss: ", ppo_loss)
        # print("Value loss: ", value_loss)

        # train nn
        self.policy_opt.zero_grad()
        self.value_opt.zero_grad()
        ppo_loss.backward()
        value_loss.backward()

        max_norm = 1.0  # Define the maximum norm value for clipping
        torch.nn.utils.clip_grad_norm_(self.policy_function.parameters(), max_norm)

        self.policy_opt.step()
        self.value_opt.step()

        # for name, param in self.policy_function.named_parameters():
        #     print(name, param.grad)

    def inference(self, obs):
        """
        """
        obs_ = self._handle_obs_dict(obs)
        action_ = self.policy_function.inference(obs_)
        action = self._action_to_dict(action_)
        return action
        


