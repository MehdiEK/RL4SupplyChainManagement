"""
Main file for training RL agent. 

Creation date: 08/03/2024
Last modification: 08/03/2024
By: Mehdi EL KANSOULI 
"""
import numpy as np
from .utils import *
import torch
from tqdm import tqdm
import torch.nn.functional as F


def normalize_obs(obs, suppliers):
    """
    """
    norm_obs = obs.copy()
    for key, val in obs.items():
        norm_obs[key] = val / np.array([suppliers[key].get("stock_max"), 150])

    return norm_obs

def PPOTrainer(agent, env, suppliers, nb_episode:int, episode_length:int, 
               nb_epochs=20):
    """
    This function train a PPO agent. 
    """
    cum_rewards = [None] * nb_episode

    for episode in range(nb_episode):

        # initialization of memory
        observations = []  # /!\ obs != states 
        actions = []
        rewards = []
        old_probs = []
        profits = []

        # initialize environment
        obs, _ = env.reset()
        norm_obs = normalize_obs(obs, suppliers=suppliers)
        observations.append(norm_obs)

        # start runnnig episode
        for _ in range(episode_length):

            # get action and save action+obs
            action = agent.get_action(norm_obs)
            actions.append(action)

            # compute old prob
            old_probs.append(agent.get_probs(norm_obs, action))

            # get new obs and save reward obtained
            obs, reward, profit = env.step(action)
            norm_obs = normalize_obs(obs, suppliers=suppliers)

            rewards.append(reward)
            profits.append(profit)
            observations.append(norm_obs)

        # print current perfomance of the agent
        cum_rewards[episode] = np.sum(rewards)
        if episode+1 % 100 == 0: 
            print("\nEpisode: ", episode)
            print("Avg daily profit: ", np.mean(profits))

        # train the agent. 
        for _ in range(nb_epochs):
            agent.train(observations, actions, rewards, old_probs)
    
    return cum_rewards 


def training_dqn(env, env_eval, model, target_network, D, actions_discrete, n_distributors, episode_length, gamma, n_steps, tau, batch_size, update_freq=1000):

    """
    :param env: environement to learn the policy
    :param env_eval: environment for evaluation
    :param model: network approximating the Q value function
    :param target_network: target network for evaluation
    :param D: replay buffer
    :param action_discrete: disretize action space
    :param n_distributors: number of distributors
    :param episode_lentgh: length of episodes
    :param gamma: dsicounted factor 
    :param n_steps: total number of timesteps to run
    :param tau: update factor for the update of the target network
    :param batch_size: batch size

    """
    store_loss = []
    reward_eval_list = []

    possible_actions = possible_actions_func(actions_discrete, n_distributors)
    n_action_values = len(actions_discrete)   

    # set up espislon for exploration
    epsilon_treshold = 0.4
    start_learning = 0

    terminated = False
    evaluation_rate = 1000

    obs, _ = env.reset()
    action_dim = n_distributors
    suppliers = ['distrib_'+str(k+1) for k in range(n_distributors)]

    for t in tqdm(range(n_steps)):
        # select action
        if t%(episode_length-1) == 0:
            terminated = True

        if terminated:
            obs, _ = env.reset()
            terminated = False

        # random action to collect data
        if t < start_learning:
            action = env.action_space.sample()

        else:  # exploration

            if np.random.random(1) <= epsilon_treshold:
                action = env.action_space.sample()
                action = [action['distrib_'+str(k+1)][0] for k in range(n_distributors)]
                action = process_sample_action(action, n_action_values-1, 200, n_distributors)
                action_ = 0
                for i in range(len(possible_actions)):
                    if action==possible_actions[i]:
                        action_ = i
                action = {'distrib_'+str(k+1): action[k] for k in range(n_distributors)}
            
            else:  # exploitation
                with torch.no_grad():
                    action_ = torch.argmax(model(handle_obs_dict(obs))).item()
                    action = action_to_dict(np.array(possible_actions[action_]), suppliers, action_dim)

        new_state, reward, profit = env.step(action)
        D.store(handle_obs_dict(obs), action_, reward, handle_obs_dict(new_state), terminated, t)
        obs = new_state

        if t >= batch_size and t % 4 == 0:
            mini_batch = D.sample(batch_size=batch_size)

            loss = 0
            criterion = F.smooth_l1_loss

            for element in mini_batch:
                if element[-2]:
                    y = torch.tensor(element[2])
                else:
                    with torch.no_grad():
                        pred_action = torch.argmax(model.forward(element[3]), dim=-1).item()
                        y = element[2] + gamma * torch.max(target_network(element[3])[0, pred_action]).detach()
                loss += criterion(model.forward(torch.tensor(element[0]))[0, element[1]], y)
                
            model.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 50)
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
            obs_eval, state_eval = env_eval.reset()
            terminated_eval = False

            for _ in range(episode_length):
                with torch.no_grad():
                    action_eval = torch.argmax(model(handle_obs_dict(obs_eval))).item()
                    action_eval = possible_actions[action_eval]
            
                obs_eval, reward_eval, profit_eval = env_eval.step({'distrib_'+str(k+1): action_eval[k] for k in range(n_distributors)})
                R += reward_eval
            reward_eval_list.append(R)
            
    return model, store_loss, reward_eval_list
