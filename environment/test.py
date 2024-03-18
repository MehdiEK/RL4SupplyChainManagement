"""
File with some functions to perform tests. 

Creation date: 27/02/2024
Last modif: 27/02/2024
By: Mehdi 
"""
from env import SupplyChainPOC, SupplyChainV0

# gloabl variables
suppliers = {
    "distrib_1": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3,
        "lost_sell": 5, 
        "transport_cost": lambda x: 10 * (x // 10 + 1), 
        "sell_price": 2
    }, 
    "distrib_2": {
        "demand": 100, 
        "stock_max": 200, 
        "stock_cost": 3,
        "lost_sell": 5, 
        "transport_cost": lambda x: 10 * (x // 10 + 1), 
        "sell_price": 2
    }  
    }


def test0(suppliers=suppliers):
    """
    Perform a test 
    """
    # demand, sotck_max, stock_cost, lost_sell, transport_cost, sell_price
    env = SupplyChainPOC(suppliers=suppliers)
    obs, state = env.reset()
    print("Initial state: ", state)
    print("Initial obs: ", obs)                 

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


def test_env_V0(suppliers=suppliers):
    """
    Perform a test 
    """
    # demand, sotck_max, stock_cost, lost_sell, transport_cost, sell_price
    env = SupplyChainV0(suppliers=suppliers)
    obs, state = env.reset()
    print("Initial state: ", state)
    print("Initial obs: ", obs)                 

    action = {
        "distrib_1": max(obs.get("distrib_1")[1] - obs.get("distrib_1")[0], 0), 
        "distrib_2": 0.
    }

    for _ in range(3):
        obs, r, profit = env.step(action)
        
        print("\nNew observation: ", obs)
        print("Reward: ", r)
        print("Profit: ", profit)



if __name__ == "__main__":
    test_env_V0()
