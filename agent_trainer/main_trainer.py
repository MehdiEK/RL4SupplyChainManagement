"""
Main file for training RL agent. 

Creation date: 08/03/2024
Last modification: 08/02/2024
By: Mehdi EL KANSOULI 
"""
import numpy as np 


def PPOTrainer(agent, env, nb_episode:int, episode_length:int):
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

        # initialize environment
        obs, _ = env.reset()

        # start runnnig episode
        for step in range(episode_length):

            # get action and save action+obs
            action = agent.get_action(obs)
            observations.append(obs)
            actions.append(action)

            # compute old prob
            old_probs.append(agent.get_probs(obs, action))

            # get new obs and save reward obtained
            obs, reward, _ = env.step(action)
            rewards.append(reward)

        # print current perfomance of the agent
        cum_rewards[episode] = np.sum(rewards)
        print("Cumulative reward: ", cum_rewards[episode])

        # train the agent. 
        agent.train(observations, actions, rewards, old_probs)
    
    return cum_rewards 


