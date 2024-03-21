import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

def create_adaptive_colored_grid_1(results, factory_char, red , blue, suppliers):
       # make a string for suppliers 1 characteristique
        distributor_1_char = f"stock max = {suppliers['distrib_1']['stock_max']}\n stock cost = {suppliers['distrib_1']['stock_cost']}\n lost sell = {suppliers['distrib_1']['lost_sell']}\n transport cost = {suppliers['distrib_1']['transport_cost']}\n sell price = {suppliers['distrib_1']['sell_price']}"
        
        for i in range(0, len(results['obs'])-1):
          
            #action string for distributor 1
            distributor_1_ac = f"\n action: {round(results['actions'][i]['distrib_1'])}"
            distributor_1_stock = f"\n actual stock = {round(results['obs'][i]['distrib_1'][0])}"
            factory_profit = f"\n profit = {round(results['profits'][i])}"
            # Initialize a white grid for 1x3   
            grid = np.ones((1, 3, 3))

            
            # Fill in the blue box on the left
            grid[0, 0] = blue

            # Fill in the red box on the right
            grid[0, 2] = red

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(grid, extent=[0, 3, 0, 1])

            # Draw the grid lines
            for x in range(4):
                ax.axvline(x=x, color='black', linewidth=2)

            # Add a horizontal line to the top and bottom
            ax.axhline(y=0, color='black', linewidth=2)
            ax.axhline(y=1, color='black', linewidth=2)
            color_profit = 'green' if round(results['profits'][i]) > 0 else 'red'

            # Factory text
            ax.text(0.5, 0.9, 'Factory', color='black', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0.5, 0.7, factory_profit, color=color_profit, ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0.5, 0.4, factory_char, color='black', ha='center', va='center', fontsize=14, fontweight='bold')


            #stock text color 
            if round(results['obs'][i]['distrib_1'][0]) == suppliers['distrib_1']['stock_max']:
                stock_color = 'red'
            else:
                 stock_color = 'green'
            # Distributor 1 text
            ax.text(2.5, 0.9, 'Distributor 1', color='black', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(2.5, 0.7, distributor_1_ac, color='green', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(2.5, 0.55, distributor_1_stock, color=stock_color, ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(2.5, 0.25, distributor_1_char, color='black', ha='center', va='center', fontsize=14, fontweight='bold')


            ax.annotate('', xy=(2.065, 0.5), xytext=(0.95, 0.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_title(f'Supply Chain Visualization day {i+1}', fontsize=14, fontweight='bold')
            plt.show()

def create_adaptive_colored_grid_2(results, factory_char, red , blue, suppliers):
    for i in range(0, len(results['obs'])-1):
            #action string for distributor 1
            distributor_1_ac = f"\n action: {round(results['actions'][i]['distrib_1'])}"
            distributor_1_stock = f"\n actual stock = {round(results['obs'][i]['distrib_1'][0])}"
            factory_profit = f"\n profit = {round(results['profits'][i])}"
            distributor_1_char = f"stock max = {suppliers['distrib_1']['stock_max']}\n stock cost = {suppliers['distrib_1']['stock_cost']}\n lost sell = {suppliers['distrib_1']['lost_sell']}\n sell price = {suppliers['distrib_1']['sell_price']}"

            distributor_2_ac = f"\n action: {round(results['actions'][i]['distrib_2'])}"
            distributor_2_stock = f"\n actual stock = {round(results['obs'][i]['distrib_2'][0])}"
            distributor_2_char = f"stock max = {suppliers['distrib_2']['stock_max']}\n stock cost = {suppliers['distrib_2']['stock_cost']}\n lost sell = {suppliers['distrib_2']['lost_sell']}\n sell price = {suppliers['distrib_2']['sell_price']}"

            # Initialize a white grid for 1x3
            grid = np.ones((3, 3, 3))


            # Fill in the blue box on the left
            grid[1, 0] = blue

            # Fill in the red box on the right
            grid[0, 2] = red

            # Fill in the red box on the right
            grid[2, 2] = red

            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(grid, extent=[0, 3, 0, 3])

            color_profit = 'green' if round(results['profits'][i]) > 0 else 'red'
            # Factory text
            ax.text(0.5, 1.9, 'Factory', color='black', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0.5, 1.7, factory_profit, color=color_profit , ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(0.5, 1.4, factory_char, color='black', ha='center', va='center', fontsize=14, fontweight='bold')

             #stock text color 
            if round(results['obs'][i]['distrib_1'][0]) == suppliers['distrib_1']['stock_max']:
                stock_color = 'red'
            else:
                 stock_color = 'green'


            # Distributor 1 text
            ax.text(2.5, 0.9, 'Distributor 1', color='black', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(2.5, 0.7, distributor_1_ac, color='green', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(2.5, 0.55, distributor_1_stock, color=stock_color, ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(2.5, 0.25, distributor_1_char, color='black', ha='center', va='center', fontsize=14, fontweight='bold')

             #stock text color 
            if round(results['obs'][i]['distrib_2'][0]) == suppliers['distrib_2']['stock_max']:
                stock_color = 'red'
            else:
                 stock_color = 'green'
            
            # Distributor 2 text
            ax.text(2.5, 2.9, 'Distributor 2', color='black', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(2.5, 2.7, distributor_2_ac, color='green', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(2.5, 2.55, distributor_2_stock, color=stock_color, ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(2.5, 2.25, distributor_2_char, color='black', ha='center', va='center', fontsize=14, fontweight='bold')

            ax.annotate('', xy=(2, 0.5), xytext=(1, 1.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
            
            ax.annotate('', xy=(2, 2.5), xytext=(1, 1.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))

            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Supply Chain Visualization day {i+1}', fontsize=14, fontweight='bold')
            plt.show()


def visualization_factory(env, results, suppliers):
    factory_char = f"prod max = {env.prod_max}\n Prod cost = {env.prod_cost}\n Excess Prod Cost = {env.excess_prod_cost}\n Excess stock cost = {env.excess_stock_cost}"

    red = [246/255, 117/255, 117/255]
    blue = [126/255, 156/255, 236/255]
    n_red_boxes = len(suppliers)
    if n_red_boxes == 1:
        create_adaptive_colored_grid_1(results, factory_char, red, blue, suppliers)

    
    if n_red_boxes == 2:
        create_adaptive_colored_grid_2(results, factory_char, red, blue, suppliers)
        
    if n_red_boxes == 3:
        # Initialize a white grid for 1x3
        grid = np.ones((5, 3, 3))

        # Set the color for red and blue boxes
        red = [1, 0, 0]
        blue = [0, 0, 1]

        # Fill in the blue box on the left
        grid[2, 0] = blue

        # Fill in the red box on the right
        grid[0, 2] = red

        # Fill in the red box on the right
        grid[2, 2] = red

        grid[4, 2] = red

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 18))
        ax.imshow(grid, extent=[0, 3, 0, 5])

        # Draw the grid lines
        for x in range(4):
            ax.axvline(x=x, color='black', linewidth=2)

        for y in range(5):
            ax.axhline(y=y, color='black', linewidth=2)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

