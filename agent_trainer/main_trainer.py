"""
Main file for training RL agent. 

Creation date: 08/03/2024
Last modification: 08/03/2024
By: Mehdi EL KANSOULI 
"""
import numpy as np 


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
        if episode % 100 == 0: 
            print("\nEpisode: ", episode)
            print("Sum of rewards: ", cum_rewards[episode])
            print("Avg profit: ", np.mean(profits))

        # train the agent. 
        for _ in range(nb_epochs):
            agent.train(observations, actions, rewards, old_probs)
    
    return cum_rewards 


