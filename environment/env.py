"""
Main file for creating environment openai gym friendly. 

Creation date: 24/02/2024
Last modification: 26/02/024
By: Mehdi EL KANSOULI 
"""

import numpy as np 
import gym 

from gym.spaces import Box, Dict


class CustomEnvPOC(gym.env):

    def __init__(self, config={}):
        """
        Initialization of the environment.
        (Transport cost is ignored for now)
        """

        # info on factory 
        self.prod_max = 1000
        self.prod_cost = 1
        # self.capex = 0  # can be ignored for now 

        # info on supplier 1
        self.demand_1 = 100
        self.stock_max_1 = 200
        self.stock_cost_1 = 5
        self.lost_sell_1 = 2

        # info on supplier 2
        self.demand_2 = 100
        self.stock_max_2 = 200
        self.stock_cost_2 = 5
        self.lost_sell_2 = 2

        # other info 
        self.step_count = 0
        self.max_steps = 10

        # define action space
        self.action_space =  Box(
            low=np.array([0., 0.]),
            high=np.array([self.prod_max, self.prod_max]), 
        )

        # define state space
        self.state_space = Dict({
            "distrib_1": Box(low=np.array(0), 
                             high=np.array(self.demand_1 + self.stock_max_1),
                             dtype=np.float32), 
            "distrib_2": Box(low=np.array(0), 
                             high=np.array(self.demand_2 + self.stock_max_2), 
                             dtype=np.float32)
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

        return initial_state


    def step(self, action):
        """
        Function to perform one step in the environment given an action. 

        :params action: int
        """
        pass