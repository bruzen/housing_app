import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

PAGE_WIDTH   = 6.3764 # thesis \the\textwidth = 460.72124pt / 72 pts_per_inch
GOLDEN_RATIO = 1.618  # (5**.5 - 1) / 2 

def set_style():
    sns.set_style("whitegrid", {'axes.grid' : True})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10']
    plt.rcParams['font.size'] = 12
    # sns.set_style("darkgrid")
    # sns.set_style("ticks",{'axes.grid' : True})

def small_multiples_lineplot(df, param_mapping, palette=None):
    set_style()

    # Desired width (thesis) and space (sns default) values
    width = PAGE_WIDTH  # inches, minus 1 inch for the legend
    space = 0.2  # replace with your desired space
    no_facets_in_height = df[param_mapping['y_global']].nunique()
    # print(no_facets_in_height)
    height       = width / GOLDEN_RATIO
    facet_height = height / ((no_facets_in_height) * (1 + space))

    # Set up the grid of subplots using FacetGrid
    g = sns.FacetGrid(df, col=param_mapping['x_global'], row=param_mapping['y_global'], 
                      margin_titles=True, height=facet_height, aspect=GOLDEN_RATIO)

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

def load_last_fast_batch_run_df():
    folder_path = os.path.join('output_data', 'data')
    files = os.listdir(folder_path)

    # OPTIONAL Filter files that start with 'results_batch_'
    files = [file for file in files if file.startswith('fast-results-batch-')]

    # Get the most recently modified file path
    most_recent_file = max(files, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))
    most_recent_file_path = os.path.join(folder_path, most_recent_file)
    # print(most_recent_file_path)
    
    return pd.read_csv(most_recent_file_path)

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