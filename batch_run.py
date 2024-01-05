import os
import yaml
import datetime
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager
from mesa.batchrunner import batch_run
from model.model import City

batch_parameters = {
            'data_collection_period': 1,
            'iterations': 1,
            'max_steps': 5
}

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
            'capital_gains_tax': 0.01, # share 0-1
        }

@contextmanager
def metadata_recorder(batch_parameters, variable_parameters, fixed_parameters, subfolder, name = None):
    metadata = {
        'experiment_name':     name,
        'git_version':         get_git_commit_hash(),
        'batch_parameters':    batch_parameters,
        'variable_parameters': variable_parameters,
        'fixed_parameters':    fixed_parameters,
        # 'simulation_parameters': model_parameters
    }

    timestamp = fixed_parameters['timestamp']
    if name:
        metadata_file_path = os.path.join(subfolder, f'metadata_batch_{timestamp}_{name}.yaml')
    else:
        metadata_file_path = os.path.join(subfolder, f'metadata_batch_{timestamp}.yaml')

    # Ensure the directory structure exists
    os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)

    # Write the new metadata to the file
    with open(metadata_file_path, 'w') as file:
        yaml.safe_dump(metadata, file)

    yield

# Define the function to run the batch simulation
def run_batch_simulation(batch_parameters, variable_parameters, model_parameters, subfolder, name = None):    
    # Run the batch simulations
    results = batch_run(City, model_parameters, **batch_parameters)
    df = pd.DataFrame(results)
    timestamp = model_parameters['timestamp']
    if name:
        data_output_path = os.path.join(subfolder, f'results_batch_{timestamp}_{name}.csv')
    else:
        data_output_path = os.path.join(subfolder, f'results_batch_{timestamp}.csv')
    df.to_csv(data_output_path, index=False)
    plot_output(df, variable_parameters, model_parameters, name)

def plot_output(df, variable_parameters, model_parameters, name = None):
    # Create the figures subfolder if it doesn't exist
    figures_folder = os.path.join('output_data', 'batch_figures')
    os.makedirs(figures_folder, exist_ok=True)

    # Define plotting styles for runs
    cmap       = plt.get_cmap('tab10')
    num_runs   = len(df['RunId'].unique())
    colors     = [cmap(i) for i in np.linspace(0, 1, num_runs)]
    linewidths = [1, 2, 3, 4]
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']  # Add more if needed
    alpha      = 0.8

    # Set the default font size for the figures
    plt.rcParams.update({'font.size': 26})

    # Create subplots with a 4x2 grid
    fig, axes = plt.subplots(4, 2, figsize=(22, 24), gridspec_kw={'hspace': 0.6})  # 4 rows, 2 columns

    # Loop through each run
    for i, run_id in enumerate(df['RunId'].unique()):
        # Subset the DataFrame for the current run and exclude time_step 0
        subset_df = df[(df['RunId'] == run_id) & (df['Step'] > 0)]

        # Extract variable parameter values for the current RunId
        variable_values = {param: subset_df[param].iloc[0] for param in variable_parameters.keys()}

        # Construct label using variable parameter values
        label = f'{", ".join(f"{key} {value}" for key, value in variable_values.items())}'

        # Use the defined styles for each run
        color = colors[i]
        linestyle = linestyles[i % len(linestyles)]  # Cycle through linestyles
        linewidth = linewidths[i % len(linewidths)]  # Cycle through linewidths

        # Plot MPL
        axes[0, 0].plot(subset_df['time_step'], subset_df['MPL'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('MPL')
        axes[0, 0].set_title(f'MPL over time')
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        # Plot n
        axes[0, 1].plot(subset_df['time_step'], subset_df['n'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('n')
        axes[0, 1].set_title(f'Urban firm workforce n over time')
        axes[0, 1].grid(True)
        axes[0, 1].legend().set_visible(False)

        # Plot N
        axes[1, 0].plot(subset_df['time_step'], subset_df['N'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('N')
        axes[1, 0].set_title(f'Total urban workforce over time')
        axes[1, 0].grid(True)
        axes[1, 0].legend().set_visible(False)

        # Plot F
        axes[1, 1].plot(subset_df['time_step'], subset_df['F'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('F')
        axes[1, 1].set_title(f'Number of firms over time')
        axes[1, 1].grid(True)
        axes[1, 1].legend().set_visible(False)

        # Plot city extent
        axes[2, 0].plot(subset_df['time_step'], subset_df['city_extent_calc'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Lot widths')
        axes[2, 0].set_title(f'Calculated city extent over time')
        axes[2, 0].grid(True)
        axes[2, 0].legend().set_visible(False)

        # Plot 'k'
        axes[2, 1].plot(subset_df['time_step'], subset_df['k'], label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].set_ylabel('k')
        axes[2, 1].set_title(f'Urban firm capital over time')
        axes[2, 1].grid(True)
        axes[2, 1].legend().set_visible(False)

        # Plot 'investor_ownership_share'
        axes[3, 0].plot(subset_df['time_step'], (1- subset_df['investor_ownership_share']), label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
        axes[3, 0].set_xlabel('Time Step')
        axes[3, 0].set_ylabel('Ownership share')
        axes[3, 0].set_title('Owner-occupier fraction over time')
        axes[3, 0].grid(True)
        axes[3, 0].legend().set_visible(False)

        # Display a single legend outside the figure
        axes[0, 0].legend(loc='center left', bbox_to_anchor=(1.15, -4.4))
        axes[3, 1].set_axis_off() 

    timestamp = model_parameters['timestamp']
    if name:
        plot_path = os.path.join(figures_folder, f'{timestamp}-{name}-timeseries-plots.pdf')
    else:
        plot_path = os.path.join(figures_folder, 'timeseries-plots.pdf')

    label_text = (
        f'\n {name} {" ".join(variable_parameters.keys())}'
        f'{plot_path}\n'
        f'adjF: {model_parameters["adjF"]}, adjw: {model_parameters["adjw"]}, '
        f'discount_rate: {model_parameters["discount_rate"]}, r_margin: {model_parameters["r_margin"]},\n'
        f'max_mortgage_share: {model_parameters["max_mortgage_share"]}, '
        f'capital_gains_tax_person: {model_parameters["capital_gains_tax_person"]}, '
        f'capital_gains_tax_investor: {model_parameters["capital_gains_tax_investor"]}'
    )

    plt.text(-1.0, -0.5, label_text, transform=plt.gca().transAxes, ha='left', va='center', wrap=True)
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
    with metadata_recorder(batch_parameters, variable_parameters, fixed_parameters, subfolder, name):
        run_batch_simulation(batch_parameters, variable_parameters, model_parameters, subfolder, name)

# Main execution
if __name__ == '__main__':
    fixed_parameters['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    subfolder = get_subfolder(fixed_parameters['timestamp'], variable_parameters)
    fixed_parameters['subfolder'] = subfolder
    model_parameters = {**fixed_parameters, **variable_parameters}
    with metadata_recorder(batch_parameters, variable_parameters, fixed_parameters, subfolder):
        run_batch_simulation(batch_parameters, variable_parameters, model_parameters, subfolder)