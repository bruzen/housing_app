import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils.file_utils as file_utils

PAGE_WIDTH   = 6.3764 # thesis \the\textwidth = 460.72124pt / 72 pts_per_inch
GOLDEN_RATIO = 1.618  # (5**.5 - 1) / 2

def set_style():
    sns.set_style("whitegrid", {'axes.grid' : True})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['cmr10']
    
    # sns.set_style("darkgrid")
    # sns.set_style("ticks",{'axes.grid' : True})

def small_multiples_lineplot(df, param_mapping, palette=None):
    set_style()
    plt.rcParams['font.size'] = 10

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

def variables_vs_time(df, variable_parameters = None):
    set_style()
    plt.rcParams['font.size'] = 10

    # df = pd.DataFrame(results)
    timestamp = df['timestamp'].iloc[0] # Same timestep for all rows in df
    figures_folder = file_utils.get_figures_subfolder()

    # TODO move to style
     # Define plotting styles for runs
    cmap       = plt.get_cmap('tab10')
    num_runs   = len(df['RunId'].unique())
    colors     = [cmap(i) for i in np.linspace(0, 1, num_runs)]
    linewidths = [.5, 1, 1.5, 2] # linewidths = [1, 2, 3, 4]
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']  # Add more if needed
    alpha      = 0.8   

    # Create subplots with a 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(.5*PAGE_WIDTH, .6*PAGE_WIDTH), gridspec_kw={'hspace': .5, 'wspace': 0.7})  # 4 rows, 2 columns

    # Loop through each run
    for i, run_id in enumerate(df['RunId'].unique()):
        # Subset the DataFrame for the current run and exclude time_step 0
        subset_df = df[(df['RunId'] == run_id) & (df['Step'] > 0)]

        if variable_parameters:
            # Extract variable parameter values for the current RunId
            variable_values = {param: subset_df[param].iloc[0] for param in variable_parameters.keys()}

            # Construct label using variable parameter values
            label = f'{", ".join(f"{key} {value}" for key, value in variable_values.items())}'
        else:
            label = None

        # Use the defined styles for each run
        color = colors[i]
        linestyle = linestyles[i % len(linestyles)]  # Cycle through linestyles
        linewidth = linewidths[i % len(linewidths)]  # Cycle through linewidths

        # Plot MPL
        axes[0, 0].plot(subset_df['time_step'], subset_df['MPL'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        # axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('MPL ($)')
        # axes[0, 0].set_title(f'MPL')
        axes[0, 0].grid(True)
        axes[0, 0].legend().set_visible(False)

        # Plot 'k'
        axes[0, 1].plot(subset_df['time_step'], subset_df['k'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        # axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('k ($)')
        # axes[0, 1].set_title(f'Firm capital') # (k)')
        axes[0, 1].grid(True)
        axes[0, 1].legend().set_visible(False)

        # Plot N
        axes[1, 0].plot(subset_df['time_step'], subset_df['N'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        # axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('N')
        # axes[1, 0].set_title(f'Total workforce') # (N)')
        axes[1, 0].grid(True)
        axes[1, 0].legend().set_visible(False)

        # Plot n
        axes[1, 1].plot(subset_df['time_step'], subset_df['n'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        # axes[0, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('n')
        # axes[0, 1].set_title(f'Firm workforce') # (n)')
        axes[1, 1].grid(True)
        axes[1, 1].legend().set_visible(False)

        # Plot city extent
        axes[2, 0].plot(subset_df['time_step'], subset_df['city_extent_calc'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        # axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Properties')
        # axes[2, 0].set_title(f'City extent')
        axes[2, 0].grid(True)
        axes[2, 0].legend().set_visible(False)

        # Plot F
        axes[2, 1].plot(subset_df['time_step'], subset_df['F'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        # axes[1, 1].set_xlabel('Time Step')
        axes[2, 1].set_ylabel('F')
        # axes[1, 1].set_title(f'Number of firms') # (F)')
        axes[2, 1].grid(True)
        axes[2, 1].legend().set_visible(False)

        # Plot 'Owner-occupier_share'
        axes[3, 0].plot(subset_df['time_step'], (1- subset_df['investor_ownership_share']), label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        axes[3, 0].set_xlabel('Time Step')
        axes[3, 0].set_ylabel('Owner-occupier \n share') #('Ownership share')
        # axes[3, 0].set_title('Owner-occupier') #('Owner-occupier fraction')
        axes[3, 0].grid(True)

        # Display a single legend outside the figure
        axes[3, 0].legend(loc='center left', bbox_to_anchor=(1.2, 0.5), frameon=False)
        axes[3, 1].set_axis_off() 

    # for ax_row in axes:
    # for ax in ax_row:
    #     ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
    #     ax.ticklabel_format(style='plain', axis='y')


    # Override font sizes
    default_font_size = plt.rcParams['font.size']
    for ax in axes.flatten():
        ax.set_title(ax.get_title(), fontsize=default_font_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=default_font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=default_font_size)

    name = f'{" ".join(variable_parameters.keys())}-{timestamp}'
    figure_filepath = file_utils.get_figures_filepath(f'{name}.pdf')
    label_text = (
        # name
        f'{timestamp}\n'
        # f'\n {name} {" ".join(variable_parameters.keys())}'
        # f'{figure_filepath}\n'
        # f'adjF: {model_parameters["adjF"]}, adjw: {model_parameters["adjw"]}, '
        # f'discount_rate: {model_parameters["discount_rate"]}, r_margin: {model_parameters["r_margin"]},\n'
        # f'max_mortgage_share: {model_parameters["max_mortgage_share"]}, '
        # f'cg_tax_per: {model_parameters["cg_tax_per"]}, '
        # f'cg_tax_invest: {model_parameters["cg_tax_invest"]}'
    )

    plt.text(-1.3, -1.3, label_text, transform=plt.gca().transAxes, ha='left', va='center', wrap=True)
    plt.savefig(figure_filepath, format='pdf', bbox_inches='tight')

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