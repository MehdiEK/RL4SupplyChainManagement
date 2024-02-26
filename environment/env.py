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

        # define state space (current stock + next day demand)
        self.state_space = Dict({
            supplier: Box(low=np.array([0., 50.]), 
                          high=np.array([val.get("stock_max"), 150]),
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

        # define initial state (actual sells the next day) 
        initial_state = self.state_space.sample()

        # define corresponding observation 
        initial_obs = self._state_to_obs(initial_state)

        # instantiate 
        self.obs = initial_obs
        self.state = initial_state

        return initial_obs, 


    def step(self, action: dict):
        """
        Function to perform one step in the environment given an action. 

        :params action: int
        """
        # intitilaize reward and total prod
        reward = 0
        total_prod = 0

        # get aciton for each supplier 
        for s, a in action.items():

            # add to total prod
            total_prod += a

            # get supplier info 
            supplier = self.suppliers.get(s)
            
            # get total nb of products 
            total_ = self.state.get(s)[0] + a

            # get new stock and demand loss (retrieve demand)
            pivot = total_ - self.state.get(s)[1]

            if pivot >= 0:
                new_stock = pivot
                demand_loss = 0
            else:
                new_stock = 0
                demand_loss = - pivot 
                        
            # compute excess stock and threshold new_stock to its limit 
            excess_stock = max(new_stock - supplier.get("stock_max"), 0) 
            new_stock = min(new_stock, supplier.get("stock_max"))

            # compute reward 
            reward_ = self._reward(supplier, new_stock, demand_loss, a, 
                                   excess_stock)

            # add to overall reward 
            reward += reward_ 

            # update state for this distributor (new stock & new demand)
            self.state[s] = np.array([new_stock, np.random.uniform(50, 150)])
        
        # compute cost of prod 
        reward += total_prod * self.prod_cost

        # add cost of excessive prod 
        reward += max(total_prod - self.prod_max, 0) * self.excess_prod_cost

        reward = self._normalize_reward(reward)  # transform in a real reward
        self.obs = self._state_to_obs(self.state)
        self.step_count += 1

        return self.obs, reward, self.step_count == self.max_steps
    
    def _normalize_reward(self, reward):
        """
        In the future, neeed to normalize the reward. Multiplication by -1 in order 
        to have a cost instead of a reaward. 
        """
        return - reward
    
    def _state_to_obs(self, state):
        """
        Noise is added to state corresonding to a non-perfect forecast of next day 
        consumption. 
        """
        obs = state.copy()
        for s, v in state.items():
            rand = np.array([0, np.random.normal(0, 5)])
            obs[s] = v + rand
        return obs
    
    def _reward(self, supplier:dict, stock, demand_loss, nb, excess):
        """
        Function to compute reward associated to exactly one distrib
        """
        # initialize reward
        reward_ = 0

        # add stock cost
        reward_ += stock * supplier.get("stock_cost")  

        # add effect of sells lost
        reward_ += demand_loss * supplier.get("lost_sell")  

        # add cost of transporting nb products
        reward_ += supplier.get("transport_cost")(nb)  

        # pay excess of stock 
        reward_ += excess * self.excess_stock_cost 

        return reward_      


def test0():
    """
    Perform a test 
    """

    # demand, sotck_max, stock_cost, lost_sell, transport_cost, sell_price

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

    env = CustomEnvPOC(suppliers=suppliers)
    initial_state = env.reset()
    print("Initial state: ", initial_state)

    for _ in range(5):

        # define dumb agent action 
        my_agent = {
            key: 100
            for key in suppliers.keys()
        }
        print("Actions: ", my_agent)
        obs, r, _ = env.step(my_agent)
        print("Obs: ", obs, r, _)
        print("State: ", env.state)
        print("\n")



if __name__ == "__main__":
    test0()
