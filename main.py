"""
Main file to run. 

The main file enables to run training of PPO and D-DQN agent and can return
the daily profit. 

For human-level agent, please run: 
    $ python main.py -human
For PPO training and evaluation:
    $ python main.py -PPO
For D-DQN: 
    $ python main.py -DQN

Creation date: 27/02/2024
Last modif: 05/04/2024
By: Mehdi 
"""

import numpy as np 
import matplotlib.pyplot as plt
import argparse

from environment.env import SupplyChainPOC, SupplyChainV0
from agent.agent import BasicAgent, PPOAgent
from agent_trainer.main_trainer import PPOTrainer, normalize_obs, training_dqn
from vis import  visualization_factory

from agent_trainer.utils import *
from agent.DQN import DQN, ReplayBuffer


def inference(env, agent, episode_length, suppliers, normalize=False):
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
        if normalize:
            action = agent.inference(normalize_obs(obs, suppliers))
        else:
            action = agent.inference(obs)

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


def main_human(episode_length=100):
    """
    Main function to run for a Human level agent. 
    """
    # define problem
    suppliers = {
    "distrib_1": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3.,
        "lost_sell": 3., 
        "transport_cost": lambda x: .5 * x, 
        "sell_price": 10.
    }, 
    "distrib_2": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3.,
        "lost_sell": 3., 
        "transport_cost": lambda x: .5 * x, 
        "sell_price": 10.
    }  
    }

    # define agent and env 
    agent = BasicAgent(suppliers)
    env = SupplyChainV0(suppliers)

    # perform an episode
    results = inference(env, agent, episode_length, suppliers)
    rewards = results.get("rewards")
    avg_profit = np.mean(results.get("profits"))

    # print("Rewards: ", rewards)
    # print("Daily profits: ", results.get("profits"))
    print("\nAvg daily profit: ", avg_profit)


def main_ppo(nb_episodes=10000, episode_length=100, nb_epochs=1):
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
        "sell_price": 10.
    }, 
    "distrib_2": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3.,
        "lost_sell": 3., 
        "transport_cost": lambda x: .5 * x, 
        "sell_price": 10.
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

    # perform a test episode
    results = inference(
        env, 
        agent, 
        episode_length=episode_length, 
        suppliers=suppliers, 
        normalize=True
    )

    # print("Rewards: ", rewards)
    # print("Daily profits: ", results.get("profits"))
    print("\nAvg daily profit: ", np.mean(results.get("profits")))

    visualization_factory(env, results, suppliers)


def mainDQN(n_distributors, n_actions=6):

    suppliers = {
        'distrib_'+str(k+1): {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3.,
        "lost_sell": 3., 
        "transport_cost": lambda x: 0.5*x, 
        "sell_price": 10.
    }
    for k in range(n_distributors)}

    action_max_value = 200
    single_possible_actions = [int(action_max_value/(n_actions-1))*i for i in range(n_actions)]

    env = SupplyChainV0(suppliers, 125*n_distributors)
    env_eval = SupplyChainV0(suppliers, 125*n_distributors)
    
    replay_buffer_capacity = 1e6
    D = ReplayBuffer(replay_buffer_capacity)

    # network parameters 
    obs_space_dim = n_distributors*2
    action_space_size = n_actions**n_distributors
    hidden_dim = 256

    model = DQN(action_space_dim=action_space_size, obs_space_dim=obs_space_dim, n_layers=2, layer_size=hidden_dim, learning_rate=3e-4)
    target_network = DQN(action_space_dim=action_space_size, obs_space_dim=obs_space_dim, n_layers=2, layer_size=hidden_dim, learning_rate=3e-4)
    target_network.load_state_dict(model.state_dict())

    trained_agent, losses, rewards = training_dqn(
        env=env,
        env_eval=env_eval,
        model=model,
        target_network=target_network,
        D=D,
        actions_discrete=single_possible_actions, 
        n_distributors=n_distributors,
        episode_length=100,
        gamma=0.9, 
        n_steps=10000, 
        tau=0.6,
        batch_size=32
    )
    
    # return trained_agent, losses, rewards
    profits = np.array(rewards) * env.prod_cost * n_distributors
    
    print("\nAvg daily profit: ", np.mean(profits))

def main():
    
    parser = argparse.ArgumentParser(description='Your script description here')

    # Add arguments
    parser.add_argument('-human', 
                        action='store_true', 
                        help='Inference of human based model')
    parser.add_argument('-PPO', 
                        action='store_true', 
                        help='Train and inference of PPO')
    parser.add_argument('-DQN', 
                        action='store_true', 
                        help='Train and inference of DQN')
    

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.human:
        main_human()
    elif args.PPO:
        main_ppo()
    elif args.DQN:
        mainDQN(n_distributors=2)
    else:
        raise ValueError("This method is not implemented yet")


if __name__ == "__main__":
    main()



