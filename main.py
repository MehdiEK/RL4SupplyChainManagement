"""
Main file to run. 

Command: $ python main.py

Creation date: 27/02/2024
Last modif: 27/02/2024
By: Mehdi 
"""

import numpy as np 
import matplotlib.pyplot as plt 

from environment.env import SupplyChainPOC, SupplyChainV0
from agent.agent import BasicAgent, PPOAgent
from agent_trainer.main_trainer import PPOTrainer, normalize_obs
from vis import  visualization_factory


def inference(env, agent, episode_length, suppliers):
    """
    Inference.

    :params env: gym env
        Supply chain env
    :params agent: gym friendly agent.
        From agent.py 
    """
    # reset env
    env.reset()

    # array of reward
    rewards = np.zeros(episode_length)
    states = [None] * (episode_length + 1)
    observations = [None] * (episode_length + 1)
    actions = [None] * episode_length
    profits = [None] * episode_length

    # start episode
    obs, state = env.reset()
    observations[0] = obs 
    states[0] = state 

    for i in range(episode_length):

        # get action from the agent
        action = agent.inference(normalize_obs(obs, suppliers))

        # perform one step 
        new_obs, reward, profit = env.step(action)

        # store result 
        rewards[i] = reward
        actions[i] = action
        observations[i+1] = new_obs
        states[i+1] = env.state
        profits[i] = profit

    results = {
        "states": states, 
        "obs": observations,
        "rewards": rewards, 
        "actions": actions, 
        "profits": profits
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
        "sell_price": 5
    }, 
    "distrib_2": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3,
        "lost_sell": 5, 
        "transport_cost": lambda x: 10 * (x // 10 + 1), 
        "sell_price": 5
    }  
    }

    # define agent and env 
    agent = BasicAgent(suppliers)
    env = SupplyChainV0(suppliers)

    # perform an episode
    results = inference(env, agent, suppliers)
    rewards = results.get("rewards")
    avg_profit = np.mean(results.get("profits"))

    print("Rewards: ", rewards)
    print("Profits: ", results.get("profits"))
    print("Avg profit: ", avg_profit)


def main_v2(nb_episodes=100, episode_length=100, nb_epochs=1):
    """
    """
    # define problem
    suppliers = {
    "distrib_1": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3.,
        "lost_sell": 3., 
        "transport_cost": lambda x: .5 * x, 
        "sell_price": 5.
    }, 
    "distrib_2": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3.,
        "lost_sell": 3., 
        "transport_cost": lambda x: .5 * x, 
        "sell_price": 5.
    }  
    }

    # define agent
    agent = PPOAgent(
        suppliers=suppliers, 
        obs_dim=4, 
        gamma=0.9
    )

    # define env 
    env = SupplyChainV0(suppliers)

    results = inference(
        env, 
        agent, 
        episode_length=episode_length, 
        suppliers=suppliers
    )

    print("Actions: ", results.get("actions"))
    print("\nObservations: ", results.get("obs"))
    print("\nReward: ", results.get("rewards"))
    print("\nProfits: ", results.get("profits"))

    rewards = PPOTrainer(
        agent=agent, 
        env=env, 
        suppliers=suppliers,
        nb_episode=nb_episodes, 
        episode_length=episode_length,
        nb_epochs=nb_epochs
    )

    plt.plot(np.arange(len(rewards)), rewards)
    plt.show()

    # print("rewards: ", rewards)
    # perform an episode
    results = inference(env, agent, episode_length=5, suppliers=suppliers)
    print("Actions: ", results.get("actions"))
    print("\nObservations: ", results.get("obs"))
    print("\nReward: ", results.get("rewards"))
    print("\nProfits: ", results.get("profits"))

    visualization_factory(env, results, suppliers)
    # test dumb agent
    # define agent and env 
    basic_agent = BasicAgent(suppliers)

    # perform an episode
    results = inference(env, basic_agent, episode_length=5, suppliers=suppliers)
    
    profit = np.sum(results.get("profits"))

    print("\nDumb agent profit: ", profit)

if __name__ == "__main__":
    main_v2(
        nb_episodes=100, 
        episode_length=50, 
        nb_epochs=1
    )



