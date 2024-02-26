"""
Main file for creating environment openai gym friendly. 

Creation date: 24/02/2024
Last modification: 26/02/024
By: Mehdi EL KANSOULI 
"""

import numpy as np 
import gym 

from gym.spaces import Box, Dict


class CustomEnvPOC(gym.Env):

    def __init__(self, suppliers: dict):
        """
        Initialization of the environment.

        :params suppliers: dict 
            Keys are suppliers, values are dictionary with keys being demand, 
            sotck_max, stock_cost, lost_sell, transport_cost, sell_price.
        """

        # info on factory 
        self.prod_max = 1000
        self.prod_cost = 1
        self.excess_prod_cost = 100
        self.excess_stock_cost = 100
        # self.capex = 0  # can be ignored for now 

        # info on suplliers 
        self.suppliers = suppliers

        self.max_steps = 100

        # define action space
        self.action_space =  Dict({
            supplier: Box(low=np.array([0]), 
                          high=np.array([self.prod_max]),
                          dtype=np.float32)
            for supplier in suppliers.keys()
        })

        # define state space
        self.state_space = Dict({
            supplier: Box(low=np.array([0]), 
                          high=np.array([val.stock_max]),
                          dtype=np.float32)
            for supplier, val in suppliers.items()
        })

        # define observation state 
        self.observation_space = self.state_space

    def _normalize_obs(self):
        """
        Function used to normalize observation to feed agent. 
        """
        pass

    def reset(self):
        """
        Define specific setting for generating an episode. 
        """
        # reset step count 
        self.step_count = 0

        # define initial state 
        initial_state = self.state_space.sample()
        self.state = initial_state

        return initial_state


    def step(self, action: dict):
        """
        Function to perform one step in the environment given an action. 

        :params action: int
        """
        # compute cost of prod 
        total_ = np.sum(action.values())
        reward = total_ * self.prod_cost

        # add cost of excessive prod 
        reward += max(total_ - self.prod_max, 0) * self.excess_prod_cost

        # get aciton for each supplier 
        for s, a in action.items():

            # get supplier info 
            supplier = self.suppliers.get(s)
            
            # get total nb of products 
            total_prod = self.state.get(s) + a

            # get new stock and demand loss (retrieve demand)
            pivot = total_prod - supplier.get("demand")

            if pivot >= 0:
                new_stock = pivot
                demand_loss = 0
            else:
                new_stock = 0
                demand_loss = - pivot 
            
            self.state[s] = np.array([new_stock])  # update state
            
            # compute excess stock 
            excess_stock = max(new_stock - supplier.get("stock_max"), 0) 

            # compute reward 
            reward_ = 0
            reward_ += new_stock * supplier.get("stock_cost")  # stock cost 
            reward_ += demand_loss * supplier.get("lost_sell")  # sell lost
            reward_ += a * supplier.get("transport_cost")  # cost of transport
            reward_ += excess_stock * self.excess_stock_cost  # pay excess stock 

            # add to overall reward 
            reward += reward_ 

        reward = self._normalize_reward(reward)  # transform in a real reward
        obs = self.state
        self.step_count += 1

        return obs, reward, self.step_count == self.max_steps
    
    def _normalize_reward(self, reward):
        """
        In the future, neeed to normalize the reward
        """
        return - reward


def test0():
    """
    Perform a test 
    """

    # demand, sotck_max, stock_cost, lost_sell, transport_cost, sell_price

    state_space = {
    "distrib_1": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3,
        "lost_sell": 5, 
        "transport_cost": 1, 
        "sell_price": 0
    }, 
    "distrib_2": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3,
        "lost_sell": 5, 
        "transport_cost": 1, 
        "sell_price": 0
    }  
    }
    
    state = state_space.sample()
    print("Initial space: ", state)

    # modif
    state["distrib_1"] = 15

    print(state)


if __name__ == "__main__":
    test0()
