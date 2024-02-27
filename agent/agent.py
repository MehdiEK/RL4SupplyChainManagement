"""
Main file for creating a RL agent. 

Creation date: 24/02/2024
Last modification: 27/02/024
By: Mehdi EL KANSOULI 
"""

class BasicAgent(object):

    def __init__(self, suppliers):
        """
        Initialization of Basic Agent. 

        :params env: gym env 
            Environment modelizing the pb
        """
        self.suppliers = suppliers 
    
    def get_action(self, observation):
        """
        Policy of the basic agent. The agent return the difference between the
        forecast for the next day and the current stock. 

        :params state: env.state object

        :return dict 
            Dictionary of actions. 
        """
        actions = {
            key: max(observation.get(key)[1] - observation.get(key)[0], 0)
            for key in self.suppliers.keys()
        }

        return actions 