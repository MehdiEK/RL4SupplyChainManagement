"""
Main file to run. 

Command: $ python main.py

Creation date: 27/02/2024
Last modif: 27/02/2024
By: Mehdi 
"""

import numpy as np 
import matplotlib.pyplot as plt 

from environment.env import SupplyChainPOC
from agent.agent import BasicAgent


def inference(env, agent, episode_length=10):
    """
    Inference.

    :params env: gym env
        Supply chain env
    :params agent: gym friendly agent.
        From agent.py 
    """
    # array of reward
    rewards = np.zeros(episode_length)
    states = [None] * (episode_length + 1)
    observations = [None] * (episode_length + 1)
    actions = [None] * episode_length

    # start episode
    obs, state = env.reset()
    observations[0] = obs 
    states[0] = state 

    for i in range(episode_length):

        # get action from the agent
        action = agent.get_action(obs)

        # perform one step 
        new_obs, reward, _ = env.step(action)

        # store result 
        rewards[i] = reward
        actions[i] = action
        observations[i+1] = new_obs
        states[i+1] = env.state

    results = {
        "states": states, 
        "obs": observations,
        "rewards": rewards, 
        "actions": actions
    }

    return results


def main():
    """
    In developpement,
    """
    # define problem
    suppliers = {
    "distrib_1": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3,
        "lost_sell": 5, 
        "transport_cost": lambda x: 10 * (x // 10 + 1), 
        "sell_price": 0
    }, 
    "distrib_2": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3,
        "lost_sell": 5, 
        "transport_cost": lambda x: 10 * (x // 10 + 1), 
        "sell_price": 0
    }  
    }

    # define agent and env 
    agent = BasicAgent(suppliers)
    env = SupplyChainPOC(suppliers)

    # perform an episode
    results = inference(env, agent)
    rewards = np.cumsum(results.get("rewards"))

    print("Rewards: ", rewards)
    plt.plot(np.arange(len(rewards)), rewards)
    plt.show()

if __name__ == "__main__":
    main()



