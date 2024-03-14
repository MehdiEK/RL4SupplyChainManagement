import torch as torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Adam 
from copy import deepcopy
from tqdm import tqdm


class ReplayBuffer():
    
    def __init__(self, capacity):
        
        self.N = capacity
        self.n_elements = 0
        self.stored_transitions = []

    def store(self, state, action, reward, new_state, terminated, timestep):

        if self.n_elements >= self.N:

            self.stored_transitions[np.argmin([transi[-1] for transi in self.stored_transitions])] = [state, action, reward, new_state, terminated, timestep]

        else:
            self.stored_transitions.append([state, action, reward, new_state, terminated, timestep])
            self.n_elements += 1


    def sample(self, batch_size):

        if self.n_elements >= batch_size:

            idx_sample = np.random.choice([i for i in range(self.n_elements)], size=batch_size)
            batch = [self.stored_transitions[i] for i in idx_sample]

            return batch
        else:

            return self.stored_transitions
        
class DQN(nn.Module):

    def __init__(self, action_space_dim, obs_space_dim, n_layers, layer_size, learning_rate):
        super().__init__()

        self.input_dim = obs_space_dim
        self.output_dim = action_space_dim
        self.hidden_dim = layer_size

        self.layers = nn.Sequential()
        self.activation = F.relu
        self.lr = learning_rate

        if n_layers == 1:

            self.layers.append(self.layers.append(nn.Linear(self.input_dim, self.output_dim)))
        
        else:


            for n in range(n_layers):
                if n == 0:

                    self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
                elif n == n_layers - 1:

                    self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))
                
                else:

                    self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        
        self.optimizer = Adam(self.parameters(), lr=1e-4)


    
    def forward(self, x):
        
        for n in range(len(self.layers)):

            x = self.layers[n](x)
            x = self.activation(x)
        
        return x
    

def training_dqn(env, env_eval, n_actions, obs_space_dim, gamma, n_steps, tau, batch_size, update_freq=1000, hidden_dim=256):

    """
    :param env: environement to learn the policy
    :param env_eval: environment for evaluation
    :param n_env: number of environment to run in parallel
    :param n_actions: number of possible actions
    :param obs_space_dim: dimension of the observation space
    :param gamma: dsicounted factor 
    :param n_steps: total number of timesteps to run
    :param tau: update factor for the update of the target network
    :param batch_size: batch size

    """


    store_loss = []
    reward_eval_list = []

    # set up espislon for exploration

    epsilon_treshold = 0.4

    replay_buffer_capacity = 1e6
    D = ReplayBuffer(replay_buffer_capacity)
    model = DQN(action_space_dim=n_actions, obs_space_dim=obs_space_dim, n_layers=2, layer_size=hidden_dim, learning_rate=3e-4)
    target_network = DQN(action_space_dim=n_actions, obs_space_dim=obs_space_dim, n_layers=2, layer_size=hidden_dim, learning_rate=3e-4)
    target_network.load_state_dict(model.state_dict())


    start_learning = 0

    state, info = env.reset()
    terminated = False
    truncated = False
    evaluation_rate = 10000

    state, info = env.reset()



    for t in tqdm(range(n_steps)):


        # select action


        if terminated or truncated:

            state, info = env.reset()

        # random action to collect data

        if t < start_learning:

            action = env.action_space.sample()

        else:

            # exploration

            if np.random.random(1) <= epsilon_treshold:

                action = env.action_space.sample()
            
            # exploitation

            else:

                with torch.no_grad():

                    action = torch.argmax(model(torch.tensor(state))).item()

        new_state, reward, terminated, truncated, info = env.step(action)

        D.store(state, action, reward, new_state, terminated, t)

        state = new_state

        if t >= batch_size and t % 4 == 0:

            mini_batch = D.sample(batch_size=batch_size)

            loss = 0
            criterion = F.smooth_l1_loss

            for element in mini_batch:

                if element[-2]:

                    y = torch.tensor(element[2])
                
                else:
            
                    with torch.no_grad():
                        pred_action = torch.argmax(model.forward(torch.tensor(element[3]))).item()
                        y = element[2] + gamma * torch.max(target_network(torch.tensor(element[3]))[pred_action]).detach()
                
                loss += criterion(model.forward(torch.tensor(element[0]))[element[1]], y)
                
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 100)

            model.optimizer.step()

            store_loss.append(loss / batch_size)
                
        if t%update_freq == 0:

            target_network_weight = target_network.state_dict()
            policy_weight = model.state_dict()

            for key in target_network_weight:

                target_network_weight[key] = tau * policy_weight[key] + (1-tau) * target_network_weight[key]
            
            target_network.load_state_dict(target_network_weight)
        
        if t%evaluation_rate == 0:

            R = 0

            obs_eval, info_eval = env_eval.reset()
            terminated_eval = False
            truncated_eval = False

            while terminated_eval == False and truncated_eval == False:
    
                with torch.no_grad():
                    action_eval = torch.argmax(model(torch.tensor(obs_eval))).item()
            
                obs_eval, reward_eval, terminated_eval, truncated_eval, info_eval = env_eval.step(action_eval)
                R += reward_eval
            reward_eval_list.append(R)


            
    return model, store_loss, reward_eval_list


def training_dqn_vectorized(env, env_eval, n_env, n_actions, obs_space_dim, gamma, n_steps, tau, batch_size):
    """
    :param env: environement to learn the policy
    :param env_eval: environment for evaluation
    :param n_env: number of environment to run in parallel
    :param n_actions: number of possible actions
    :param obs_space_dim: dimension of the observation space
    :param gamma: dsicounted factor 
    :param n_steps: total number of timesteps to run
    :param tau: update factor for the update of the target network
    :param batch_size: batch size

    """
    

    store_loss = []
    reward_eval_list = []

    # set up espislon for exploration

    epsilon_treshold = 0.4

    replay_buffer_capacity = 1e6
    D = ReplayBuffer(replay_buffer_capacity)
    model = DQN(action_space_dim=n_actions, obs_space_dim=obs_space_dim, n_layers=2, layer_size=128, learning_rate=3e-4)
    target_network = DQN(action_space_dim=n_actions, obs_space_dim=obs_space_dim, n_layers=2, layer_size=128, learning_rate=4e-4)
    target_network.load_state_dict(model.state_dict())


    update_freq = 1000
    start_learning = 0

    state, info = env.reset()
    terminated = np.array([False for _ in range(n_env)])
    truncated = np.array([False for _ in range(n_env)])
    evaluation_rate = 10000

    state, info = env.reset()



    for t in tqdm(range(n_steps)):


        # select action

        if any(terminated) or any(truncated):

            state, info = env.reset()

        # random action to collect data

        if t < start_learning:

            action = env.action_space.sample()

        else:

            # exploration

            if np.random.random(1) <= epsilon_treshold:

                action = env.action_space.sample()
            
            # exploitation

            else:

                with torch.no_grad():

                    action = torch.argmax(model(torch.tensor(state)), dim=1).numpy()
        


        new_state, reward, terminated, truncated, info = env.step(action)

        for s, a, r, ns, term in zip(state, action, reward, new_state, terminated):

            D.store(torch.tensor(s), torch.tensor([a]), torch.tensor([r]), torch.tensor(ns), [term], t)

        state = new_state

        if t >= batch_size and t % 4 == 0:

            mini_batch = D.sample(batch_size=batch_size)

            loss = 0
            criterion = F.smooth_l1_loss

            actions = torch.cat([e[1].reshape((1, 1)) for e in mini_batch], dim=0)
            states = torch.cat([e[0].reshape((1, obs_space_dim)) for e in mini_batch], dim=0)
            rewards = torch.cat([e[2].reshape((1, 1)) for e in mini_batch], dim=0)
            next_states = torch.cat([e[3].reshape((1, obs_space_dim)) for e in mini_batch], dim=0)
            terminates = np.concatenate([e[4] for e in mini_batch])


            with torch.no_grad():
        
                y = rewards + gamma * torch.max(target_network(next_states), axis=1).values.reshape(batch_size, 1) * torch.tensor(1-terminates).reshape(batch_size, 1)
            
            loss += criterion(
                model.forward(states).gather(1, actions).reshape_as(y),
                y
                )
                
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 100)

            model.optimizer.step()

            store_loss.append(loss / batch_size)
                
        if t%update_freq == 0:

            target_network_weight = target_network.state_dict()
            policy_weight = model.state_dict()

            for key in target_network_weight:

                target_network_weight[key] = tau * policy_weight[key] + (1-tau) * target_network_weight[key]
            
            target_network.load_state_dict(target_network_weight)
        
        if t%evaluation_rate == 0 and t!=0:

            R = 0

            obs_eval, info_eval = env_eval.reset()
            terminated_eval = np.array([False for _ in range(n_env)])
            truncated_eval = np.array([False for _ in range(n_env)])

            while any(terminated_eval) == False and any(truncated_eval) == False:
    
                with torch.no_grad():
                    action_eval = torch.argmax(model(torch.tensor(obs_eval)), dim=1).numpy()
            
                obs_eval, reward_eval, terminated_eval, truncated_eval, info_eval = env_eval.step(action_eval)
                R += reward_eval
            reward_eval_list.append(R)


            
    return model, store_loss, reward_eval_list