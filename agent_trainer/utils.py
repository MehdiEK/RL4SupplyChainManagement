import numpy as np
import torch


def possible_actions_func(single_possible_actions, n_distributors):

    actions = [[s] for s in single_possible_actions]

    for i in range(n_distributors-1):

        actions = np.array([[s + [t] for s in actions] for t in single_possible_actions])
        actions = actions.reshape(-1, i+2).tolist()
    
    return actions

def handle_obs_dict(obs):
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

def action_to_dict(action, suppliers, action_dim):
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
    for i in range(action_dim):
        actions_dict[suppliers[i]] = action[i].item()
    
    action_ = actions_dict  # Dict(actions_dict)
    return action_

def process_sample_action(action: list, n_actions, max_action_val, n_distributors):

    processed_action = [None for _ in range(n_distributors)]

    for j in range(n_distributors):

        if action[j] > 200:
            processed_action[j] = 200


    for i in range(n_actions):
        for j in range(n_distributors):

            if action[j] < max_action_val/n_actions * (i+1) and action[j] >= max_action_val/n_actions * (i):

                processed_action[j] = max_action_val/n_actions * (i)

    
    return processed_action

def moving_average(array, window):
    means = np.zeros(len(array) - window, dtype=np.float32)
    for i in range(len(array) - window):
        subarray = array[i:i+window]
        avg = np.sum(subarray)
        means[i] = avg
    return [np.mean(means), np.std(means)]