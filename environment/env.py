"""
Main file for creating environment openai gym friendly. 

Creation date: 24/02/2024
Last modification: 26/02/024
By: Mehdi EL KANSOULI 
"""

import numpy as np 
import gym 

from gym.spaces import Box, Dict


class SupplyChainPOC(gym.Env):

    def __init__(self, suppliers: dict):
        """
        Initialization of the environment.

        :params suppliers: dict 
            Keys are suppliers, values are dictionary with keys being demand, 
            sotck_max, stock_cost, lost_sell, transport_cost, sell_price.
        """

        # info on factory 
        self.prod_max = 500
        self.prod_cost = 3.
        self.excess_prod_cost = 2.
        self.excess_stock_cost = 10.
        self.sell_price = 10.

        # info on supliers 
        self.suppliers = suppliers
        self.nb_suppliers = len(suppliers.keys())

        # define action space
        self.action_space =  Dict({
            supplier: Box(low=np.array([0.]), 
                          high=np.array([self.prod_max]),
                          dtype=np.float64)
            for supplier in suppliers.keys()
        })

        # define state space (current stock + next day demand)
        self.state_space = Dict({
            supplier: Box(low=np.array([0., 50.]), 
                          high=np.array([val.get("stock_max"), 150]),
                          dtype=np.float64)
            for supplier, val in suppliers.items()
        })

        # define observation state 
        self.observation_space = self.state_space

    def reset(self):
        """
        Define specific setting for generating an episode. 
        """
        # define initial state (actual sells the next day) 
        initial_state = self.state_space.sample()

        # define corresponding observation 
        initial_obs = self._state_to_obs(initial_state)

        # instantiate 
        self.obs = initial_obs
        self.state = initial_state

        return self._normalize_obs(initial_obs), initial_state


    def step(self, action: dict):
        """
        Function to perform one step in the environment given an action. 

        :params action: int
        """
        # intitilaize reward and total prod
        reward = 0
        total_prod = 0
        penalties = 0
        action = self._denormalize_actions(action)
        if np.random.random() <= 0.001:
            print("Action: ", action)

        # get aciton for each supplier 
        for s, a in action.items():

            if a < 0:
                # raise ValueError("Production cannot be negative")
                penalties += 10
                a = 0

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
                reward -= self.state.get(s)[1] * self.sell_price
            else:
                new_stock = 0
                demand_loss = - pivot 
                reward -= total_ * self.sell_price
                reward += supplier.get("lost_sell") * demand_loss
                        
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

        # add penalties (e.g. pred neg actions)
        profit = - reward
        reward += penalties

        reward = self._normalize_reward(reward)  # transform in a real reward
        self.obs = self._state_to_obs(self.state)

        return self._normalize_obs(self.obs), reward, profit
    
    def _normalize_reward(self, reward):
        """
        Normalize the reward. Multiplication by -1 in order 
        to have a cost instead of a reaward. 
        """
        return - reward / (self.prod_cost * 100 * len(self.suppliers))
    
    def _normalize_obs(self, obs):
        """
        """
        norm_obs = []
        for distrib, arr in obs.items():
            x, y = arr.copy()
            x /= self.suppliers.get(distrib).get("stock_max")
            y /= (self.prod_max / len(self.suppliers.keys()))
            norm_obs += [x, y]
        return np.array(norm_obs)
            
    def _denormalize_actions(self, actions):
        """
        Function that takes as input normalize actions and denormalize 
        them

        :params actions: dict 
            Dictionary of normalized actions 
        
        :return dict 
            Denormalized actions 
        """
        denorm_action = {}
        for key, value in actions.items():
            real_action = value * (self.prod_max / len(self.suppliers.keys()))
            denorm_action[key] = real_action

        return denorm_action
    
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
    

class SupplyChainV0(gym.Env):
    
    def __init__(self, suppliers: dict):
        """
        Initialization of the environment.

        :params suppliers: dict 
            Keys are suppliers, values are dictionary with keys being demand, 
            sotck_max, stock_cost, lost_sell, transport_cost, sell_price.
        """
        # info on factory 
        self.prod_max = 250
        self.prod_cost = 1.
        self.excess_prod_cost = 2.
        self.excess_stock_cost = 5.

        # info on supliers 
        self.suppliers = suppliers
        self.nb_suppliers = len(suppliers.keys())

        # define action space
        self.action_space =  Dict({
            supplier: Box(low=np.array([0.]), 
                          high=np.array([self.prod_max]),
                          dtype=np.float64)
            for supplier in suppliers.keys()
        })

        # define state space (current stock + next day demand)
        self.state_space = Dict({
            supplier: Box(low=np.array([0., 50.]), 
                          high=np.array([val.get("stock_max"), 150]),
                          dtype=np.float64)
            for supplier, val in suppliers.items()
        })

        # define observation state 
        self.observation_space = self.state_space

    def reset(self):
        """
        Define specific setting for generating an episode. 
        """
        # define initial state (actual sells the next day) 
        initial_state = self.state_space.sample()

        # define corresponding observation 
        initial_obs = self._state_to_obs(initial_state)

        # instantiate 
        self.obs = initial_obs
        self.state = initial_state

        return initial_obs, initial_state

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

            if a < 0:
                raise ValueError("Production cannot be negative")

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
                # compute earnings
                reward -= self.state.get(s)[1] * supplier.get("sell_price", 0.)
            else:
                new_stock = 0
                demand_loss = - pivot 
                # compute earnings
                reward -= total_ * supplier.get("sell_price", 0.)
                        
            # compute excess stock and threshold new_stock to its limit 
            excess_stock = max(new_stock - supplier.get("stock_max"), 0) 
            new_stock = min(new_stock, supplier.get("stock_max"))

            # compute reward 
            costs = self._all_costs(supplier, new_stock, demand_loss, a, 
                                    excess_stock)

            # add to overall reward 
            reward += costs

            # update state for this distributor (new stock & new demand)
            self.state[s] = np.array([new_stock, np.random.uniform(50, 150)])
        
        # compute cost of prod 
        reward += total_prod * self.prod_cost

        # add cost of excessive prod 
        reward += max(total_prod - self.prod_max, 0) * self.excess_prod_cost

        # add penalties (e.g. pred neg actions)
        reward *= -1
        profit = reward

        # normalize the reward and update observation
        reward = self._normalize_reward(reward)  # transform in a real reward
        self.obs = self._state_to_obs(self.state)

        return self.obs, reward, profit
    
    def _normalize_reward(self, reward):
        """
        Normalize the reward. Multiplication by -1 in order 
        to have a cost instead of a reaward. 
        """
        return reward / 100
    
    def _state_to_obs(self, state):
        """
        Noise is added to state corresonding to a non-perfect forecast of next day 
        consumption. 
        """
        obs = state.copy()
        for s, v in state.items():
            stock, next_cons = v
            forecast = np.clip(next_cons + np.random.normal(0, 10), 50, 150)
            obs[s] = np.array([stock, forecast])
        return obs
    
    def _all_costs(self, supplier:dict, stock, demand_loss, nb, excess):
        """
        Function to compute reward associated to exactly one distrib
        """
        # initialize reward
        cost = 0

        # add stock cost
        cost += stock * supplier.get("stock_cost")  

        # add effect of sells lost
        cost += demand_loss * supplier.get("lost_sell")  

        # add cost of transporting nb products
        cost += supplier.get("transport_cost")(nb)  

        # pay excess of stock 
        cost += excess * self.excess_stock_cost 

        return cost      


