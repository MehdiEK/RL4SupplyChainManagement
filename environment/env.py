"""
Main file for creating environment openai gym friendly. 

Creation date: 24/02/2024
Last modification: 24/02/024
By: Mehdi EL KANSOULI 
"""

import gym 


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
        self.count = 0
        self.max_steps = 10

        # define action space

        # define observation space

    def step(self, action):
        """
        Function to perform one step in the environment given an action. 

        :params action: int
        """
        pass