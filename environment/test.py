"""
File with some functions to perform tests. 

Creation date: 27/02/2024
Last modif: 27/02/2024
By: Mehdi 
"""
from env import SupplyChainPOC

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

    env = SupplyChainPOC(suppliers=suppliers)
    obs, state = env.reset()
    print("Initial state: ", state)

    for _ in range(5):

        # define dumb agent action 
        my_agent = {
            key: max(obs.get(key)[1] - obs.get(key)[0], 0)
            for key in suppliers.keys()
        }
        print("Actions: ", my_agent)
        obs, r, _ = env.step(my_agent)
        print("Obs: ", obs, r)
        print("State: ", env.state)
        print("\n")



if __name__ == "__main__":
    test0()
