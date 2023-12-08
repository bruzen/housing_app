import os
import sys
import yaml
import datetime
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from contextlib import contextmanager
from mesa.batchrunner import batch_run
from model.model import City

# Define the variable and fixed parameters
variable_parameters = {
    'density': [1, 100],
    # 'subsistence_wage': [10000, 30000],
    'gamma': [0.02]
}

fixed_parameters = {
            'run_notes': 'Debugging model.',
            'subfolder': None,
            'width':     10, #30,
            'height':    10, #30,

            # FLAGS
            'demographics_on': True,  # Set flag to False for debugging to check firm behaviour without demographics or housing market
            'center_city':     False, # Flag for city center in center if True, or bottom corner if False
            # 'random_init_age': False,  # Flag for randomizing initial age. If False, all workers begin at age 0
            'random_init_age': True,  # Flag for randomizing initial age. If False, all workers begin at age 0

            # LABOUR MARKET AND FIRM PARAMETERS
            'subsistence_wage': 40000., # psi
            'init_city_extent': 10.,    # CUT OR CHANGE?
            'seed_population': 400,
            'init_wage_premium_ratio': 0.2, # 1.2, ###

            # PARAMETERS MOST LIKELY TO AFFECT SCALE
            'c': 300.0,                            ###
            'price_of_output': 10,                 ######
            'density':600,                         #####
            'A': 3000,                             ### 
            'alpha': 0.18,
            'beta':  0.75,
            'gamma': 0.12, ### reduced from .14
            'overhead': 1,
            'mult': 1.2,
            'adjN': 0.15,
            'adjk': 0.10,
            'adjn': 0.25,
            'adjF': 0.15,
            'adjw': 0.02, 
            'dist': 1, 
            'init_F': 100.0,
            'init_k': 500.0,
            'init_n': 100.0,

            # HOUSING AND MORTGAGE MARKET PARAMETERS
            'mortgage_period': 5.0,       # T, in years
            'working_periods': 40,        # in years
            'savings_rate': 0.3,
            'discount_rate': 0.07,        # 1/delta
            'r_prime': 0.05,
            'r_margin': 0.01,
            'property_tax_rate': 0.04,     # tau, annual rate, was c
            'housing_services_share': 0.3, # a
            'maintenance_share': 0.2,      # b
            'max_mortgage_share': 0.9,
            'ability_to_carry_mortgage': 0.28,
            'wealth_sensitivity': 0.1,
        }

batch_parameters = {
            'data_collection_period': 1,
            'iterations': 1,
            'max_steps': 5
}

@contextmanager
def metadata_recorder(model_parameters, batch_parameters, subfolder, name = None):
    metadata = {
        'experiment_name':  name,
        'batch_parameters':     batch_parameters,
        'git_version':           get_git_commit_hash(),
        'simulation_parameters': model_parameters
    }

    metadata_file_path = os.path.join(subfolder, f'metadata_batch.yaml')
    # Check if the file exists
    file_exists = os.path.isfile(metadata_file_path)

    # If the file exists, load the existing metadata; otherwise, create an empty dictionary
    if file_exists:
        with open(metadata_file_path, 'r') as file:
            existing_metadata = yaml.safe_load(file)
    else:
        existing_metadata = {}

    # Append the metadata for the current experiment to the existing metadata dictionary
    existing_metadata[name] = metadata

    # Write the updated metadata back to the file
    with open(metadata_file_path, 'w') as file:
        yaml.safe_dump(existing_metadata, file)

    yield

# Define the function to run the batch simulation
def run_batch_simulation(batch_parameters, variable_parameters, model_parameters, subfolder, name = None):    
    # Run the batch simulations
    results = batch_run(City, model_parameters, **batch_parameters)
    df = pd.DataFrame(results)
    if name:
        data_output_path = os.path.join(subfolder, f'{name}_batch_results.csv')
    else:
        data_output_path = os.path.join(subfolder, f'batch_results.csv')
    df.to_csv(data_output_path, index=False)
    plot_output(df, variable_parameters, subfolder, name)

def plot_output(df, variable_parameters, subfolder, name = None):
    # Create the figures subfolder if it doesn't exist
    figures_folder = os.path.join(subfolder, 'figures')
    os.makedirs(figures_folder, exist_ok=True)

    # Define a color map for runs
    cmap = plt.get_cmap('tab10')
    num_runs = len(df['RunId'].unique())
    colors = [cmap(i) for i in np.linspace(0, 1, num_runs)]

    # Plot ownership
    # Create plot
    plt.figure(figsize=(10, 6))

    for i, run_id in enumerate(df['RunId'].unique()):
        # subset_df = df[df['RunId'] == run_id]
        subset_df = df[(df['RunId'] == run_id) & (df['Step'] > 0)]  # Exclude time_step 0

        # Extract variable parameter values for the current RunId
        variable_values = {param: subset_df[param].iloc[0] for param in variable_parameters.keys()}
        # print(variable_values)
        # print {str(variable_values)}
        
        # Construct label using variable parameter values
        label = f'{", ".join(f"{key} {value}" for key, value in variable_values.items())}'

        # Use the defined color for each run
        color = colors[i]

        plt.plot(subset_df['time_step'], subset_df['investor_ownership_share'], label=label, linestyle='-', color=color)

    plt.xlabel('Time Step')
    plt.ylabel('Ownership share')
    plt.title('Ownership share')
    plt.legend()

    # Save the line plot to the figures subfolder
    if name:
        plot_path = os.path.join(figures_folder, f'{name}_warranted_price_vs_time_step.pdf')
    else:
        plot_path = os.path.join(figures_folder, 'warranted_price_vs_time_step.pdf')
    plt.text(0.5, -0.12, plot_path, transform=plt.gca().transAxes, ha='center', va='center', fontsize=7)
    plt.savefig(plot_path, format='pdf')

    # Plot other variables
    # Create subplots with a 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(15, 15))  # 4 rows, 2 columns
    # Adjust subplot spacing
    fig.subplots_adjust(hspace=0.5, wspace=0.3)

    # Set the title for the grid
    fig.suptitle(f'{name} {" ".join(variable_parameters.keys())}', fontsize=16)

    # Loop through each run
    for i, run_id in enumerate(df['RunId'].unique()):
        # Subset the DataFrame for the current run and exclude time_step 0
        subset_df = df[(df['RunId'] == run_id) & (df['Step'] > 0)]

        # Extract variable parameter values for the current RunId
        variable_values = {param: subset_df[param].iloc[0] for param in variable_parameters.keys()}

        # Construct label using variable parameter values
        label = f'{", ".join(f"{key} {value}" for key, value in variable_values.items())}'

        # Use the defined color for each run
        color = colors[i]

        # # Determine the subplot position based on run_id
        # row_position = (run_id - 1) // 2  # row position (0-3)
        # col_position = (run_id - 1) % 2  # column position (0 or 1)

        # Plot MPL
        axes[0, 0].plot(subset_df['time_step'], subset_df['MPL'], label=label, color=color)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('MPL')
        axes[0, 0].set_title(f'MPL over time')
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        # Plot n
        axes[0, 1].plot(subset_df['time_step'], subset_df['n'], label=label, color=color)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('n')
        axes[0, 1].set_title(f'Urban firm workforce n over time')
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        # Plot N
        axes[1, 0].plot(subset_df['time_step'], subset_df['N'], label=label, color=color)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('N')
        axes[1, 0].set_title(f'Total urban workforce over time')
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        # Plot F
        axes[1, 1].plot(subset_df['time_step'], subset_df['F'], label=label, color=color)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('F')
        axes[1, 1].set_title(f'Number of firms over time')
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        # Plot city extent
        axes[2, 0].plot(subset_df['time_step'], subset_df['city_extent_calc'], label=label, color=color)
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Lot widths')
        axes[2, 0].set_title(f'City extent over time')
        axes[2, 0].grid(True)
        axes[2, 0].legend()

        # Plot N/F
        axes[2, 1].plot(subset_df['time_step'], subset_df['N']/subset_df['F'], label=label, color=color)
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].set_ylabel('N/F')
        axes[2, 1].set_title(f'Workforce divided by number of firms over time')
        axes[2, 1].grid(True)
        axes[2, 1].legend()

        # Plot 'investor_ownership_share'
        axes[3, 0].plot(subset_df['time_step'], subset_df['investor_ownership_share'], label=label, color=color)
        axes[3, 0].set_xlabel('Time Step')
        axes[3, 0].set_ylabel('Ownership share')
        axes[3, 0].set_title(f'Ownership share over time')
        axes[3, 0].grid(True)
        axes[3, 0].legend()

        # Plot 'k'
        axes[3, 1].plot(subset_df['time_step'], subset_df['k'], label=label, color=color)
        axes[3, 1].set_xlabel('Time Step')
        axes[3, 1].set_ylabel('k')
        axes[3, 1].set_title(f'Urban firm capital over time')
        axes[3, 1].grid(True)
        axes[3, 1].legend()

    if name:
        plot_path = os.path.join(figures_folder, f'{name}_timeseries_plots.pdf')
    else:
        plot_path = os.path.join(figures_folder, 'timeseries_plots.pdf')
    plt.text(0.5, -0.3, plot_path, transform=plt.gca().transAxes, ha='center', va='center')
    plt.savefig(plot_path, format='pdf')

def get_subfolder(timestamp, variable_parameters = None):
    # Name is used in subfolder name if variable_parameters are not passed
    # Create the subfolder path
    output_data_folder = 'output_data'
    runs_folder = 'batch_runs'
    # if variable_parameters:
    #     parameter_names = '-'.join(variable_parameters.keys())
    #     subfolder = os.path.join(output_data_folder, runs_folder, f"{timestamp}--{parameter_names}")
    # else:
    subfolder = os.path.join(output_data_folder, runs_folder, f"{timestamp}")

    # Create the subfolder if it doesn't exist
    os.makedirs(subfolder, exist_ok=True)

    return subfolder

def get_git_commit_hash():
    try:
        # Run 'git rev-parse HEAD' to get the commit hash
        result = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return result.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None

def run_experiment(batch_parameters, variable_parameters, fixed_parameters, name = None):
    subfolder = get_subfolder(fixed_parameters['timestamp'])
    fixed_parameters['subfolder'] = subfolder
    model_parameters = {**fixed_parameters, **variable_parameters}
    with metadata_recorder(model_parameters, batch_parameters, subfolder, name):
        run_batch_simulation(batch_parameters, variable_parameters, model_parameters, subfolder, name)

# Main execution
if __name__ == '__main__':
    fixed_parameters['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    subfolder = get_subfolder(fixed_parameters['timestamp'], variable_parameters)
    fixed_parameters['subfolder'] = subfolder
    model_parameters = {**fixed_parameters, **variable_parameters}
    with metadata_recorder(model_parameters, batch_parameters, subfolder):
        run_batch_simulation(batch_parameters, variable_parameters, model_parameters, subfolder)