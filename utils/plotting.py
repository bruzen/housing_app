import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def small_multiples_lineplot(df, param_mapping, palette=None):
    # Set up the grid of subplots using FacetGrid
    g = sns.FacetGrid(df, col=param_mapping['x_global'], row=param_mapping['y_global'], 
                      margin_titles=True, height=2.5, aspect=1.)

    # Add grey gridlines (note: only shows up on second run)
    # sns.set_style("darkgrid")
    # sns.set_style("ticks",{'axes.grid' : True})
    sns.set_style("whitegrid", {'axes.grid' : True})

    # sns.set(font_scale=1.5)
    # sns.set_context("paper", rc={"font.size":20,"axes.titlesize":20,"axes.labelsize":20})   
    # sns.set(rc={'figure.figsize':(11.7,8.27),"font.size":20,"axes.titlesize":20,"axes.labelsize":20},style="white")

    if palette:
        g.map_dataframe(sns.lineplot, x=param_mapping['x'], y=param_mapping['y'], estimator=None, 
                        hue=param_mapping['line_val'], palette=palette).add_legend() #(fontsize=20)
    else:
        g.map_dataframe(sns.lineplot, x=param_mapping['x'], y=param_mapping['y'], estimator=None, 
                        hue=param_mapping['line_val']).add_legend() # (fontsize=20) , marker='o'

    # Set axis labels and titles for each subplot
    g.set_axis_labels(format_label(param_mapping['x']), format_label(param_mapping['y'])).set_titles(
        row_template=f'{format_label(param_mapping["y_global"])} = {{row_name}}', 
        col_template=f'{format_label(param_mapping["x_global"])} = {{col_name}}'
    )

    # Show the plot
    plt.show()

def format_label(label):
    # Capitalize the first letter of each word
    return ' '.join(word.capitalize() for word in label.split('_'))

def downsample(df, var, no_vals_to_plot):
    # Get unique values for the variable
    unique_time_steps = df[var].unique()

    # Sample indices
    sampled_indices = np.linspace(0, len(unique_time_steps) - 1, no_vals_to_plot, dtype=int)

    # Use the sampled indices to get sampled values
    sampled_values = unique_time_steps[sampled_indices]

    # Filter the DataFrame based on the sampled values
    return df[df[var].isin(sampled_values)]

def get_bidder_color_palette(bidder_categories):
    numeric_values = [int(category.split()[-1]) for category in bidder_categories if category.startswith('Savings')]
    min_value = min(numeric_values) if numeric_values else 0
    max_value = max(numeric_values) if numeric_values else 1
    brightness_factor = 0.15
    compression_factor = 0.4

    color_palette = {
        # category: 'black' if category == 'Investor' else sns.light_palette('blue', as_cmap=True)(int(category.split()[-1]) / 480000) # GOOD
        category: 'black' if category == 'Investor' else sns.light_palette('blue', as_cmap=True)((int(category.split()[-1]) - min_value) * compression_factor / (max_value - min_value) + brightness_factor)
        for category in bidder_categories
    }

    return color_palette

    # # Could also hard code colors
    # color_palette = {
    #     'savings_1': 'lightblue', 
    #     'savings_2': 'skyblue', 
    #     'savings_3': 'deepskyblue',   
    #     'investor': 'black'
    # }

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
    
    # Plot dist vs bid on each subplot
    parameter_mapping = {
        'x': 'dist',
        'y': 'bid',
        'x_global': 'param_1',
        'y_global': 'time_step',
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