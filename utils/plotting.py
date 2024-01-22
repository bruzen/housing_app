import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def small_multiples_lineplot(df, param_mapping, palette=None):
    # Set up the grid of subplots using FacetGrid
    g = sns.FacetGrid(df, col=param_mapping['x_global'], row=param_mapping['y_global'], 
                      margin_titles=True, height=3, aspect=1.5)

    # Add grey gridlines (note: only shows up on second run)
    # sns.set_style("darkgrid")
    # sns.set_style("ticks",{'axes.grid' : True})
    sns.set_style("whitegrid", {'axes.grid' : True})

    if palette:
        g.map_dataframe(sns.lineplot, x=param_mapping['x'], y=param_mapping['y'], 
                        hue=param_mapping['line_val'], marker='o', palette=palette).add_legend()
    else:
        g.map_dataframe(sns.lineplot, x=param_mapping['x'], y=param_mapping['y'], 
                        hue=param_mapping['line_val'], marker='o').add_legend()

    # Set axis labels and titles for each subplot
    g.set_axis_labels(param_mapping['x'], param_mapping['y']).set_titles(
        row_template=f'{format_label(param_mapping["y_global"])} = {{row_name}}', 
        col_template=f'{param_mapping["x_global"]} = {{col_name}}'
    )

    # Show the plot
    plt.show()

def format_label(label):
    # Capitalize the first letter of each word
    return ' '.join(word.capitalize() for word in label.split('_'))

if __name__ == "__main__":

    # Generate example dataframe
    no_param_values = 2
    no_distances    = 3
    no_time_steps   = 2
    agent_names     = ['investor', 'savings_1', 'savings_2', 'savings_3']
    data            = []

    for param_1 in range(no_param_values):
        for dist in range(no_distances):
            for time_step in range(no_time_steps):
                for agent_no, agent_name in enumerate(agent_names):
                    bid = agent_no * (time_step * 10 + dist * 30) + 100 * param_1

                    data.append([dist, bid, param_1, time_step, agent_name])

    df = pd.DataFrame(data, columns=['dist', 'bid', 'param_1', 'time_step', 'agent_name'])

    # Determine which df columns go on which axes
    
    # Plot dist vs bid on each subplot
    # parameter_mapping = {
    #     'x': 'dist',
    #     'y': 'bid',
    #     'x_global': 'param_1',
    #     'y_global': 'time_step',
    #     'line_val': 'agent_name'
    # }

    # Plot time_step vs bid on each subplot
    parameter_mapping = {
        'x': 'time_step',
        'y': 'bid',
        'x_global': 'param_1',
        'y_global': 'dist',
        'line_val': 'agent_name'
    }

    # Set the color palette
    color_palette = {
        'savings_1': 'lightblue', 
        'savings_2': 'skyblue', 
        'savings_3': 'deepskyblue', 
        'investor': 'black'
    }

    # Plot the data
    small_multiples_lineplot(df, parameter_mapping, palette=color_palette)