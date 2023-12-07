import os
import sys
import yaml
import datetime
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
    # 'gamma': [0.001, 0.02, 0.7]
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

# Define the context manager to record metadata
@contextmanager
def metadata_recorder(model_parameters, batch_parameters, subfolder):
    metadata = {
        'model_parameters': model_parameters,
        'batch_parameters': batch_parameters
    }
    yield metadata
    # Save the metadata to a YAML file
    metadata_path = os.path.join(subfolder, 'batch_metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata, f)

# Define the function to run the batch simulation
def run_batch_simulation(model_parameters, batch_parameters, subfolder):    
    # Run the batch simulations
    results = batch_run(City, model_parameters, **batch_parameters)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(subfolder, f'batch_results.csv'), index=False)
    
    # Create the figures subfolder if it doesn't exist
    figures_folder = os.path.join(subfolder, 'figures')
    os.makedirs(figures_folder, exist_ok=True)

    # Create a line plot
    plt.figure(figsize=(10, 6))

    for run_id in df['RunId'].unique():
        # subset_df = df[df['RunId'] == run_id]
        subset_df = df[(df['RunId'] == run_id) & (df['Step'] > 0)]  # Exclude time_step 0

        # Extract variable parameter values for the current RunId
        variable_values = {param: subset_df[param].iloc[0] for param in variable_parameters.keys()}
        
        # Construct label using variable parameter values
        label = f'Run {run_id}: {", ".join(f"{key} {value}" for key, value in variable_values.items())}'
    
        plt.plot(subset_df['time_step'], subset_df['wage_premium'], label=label, linestyle='-')

    plt.xlabel('Time Step')
    plt.ylabel('$')
    plt.title('Wage Premium')
    plt.legend()

    # Save the line plot to the figures subfolder
    plot_path = os.path.join(figures_folder, 'warranted_price_vs_time_step.png')
    plt.savefig(plot_path)

    # # Create a set of heatmaps
    # run_id_1_df = df[df['RunId'] == 1]
    # create_heatmap(run_id_1_df, 'is_working', [[0, 'blue'], [1, 'red']], output_folder=figures_folder)

def get_subfolder(timestamp, variable_parameters = None, name = None):
    # Name is used in subfolder name if variable_parameters are not passed
    # Create the subfolder path
    output_data_folder = 'output_data'
    runs_folder = 'batch_runs'
    if variable_parameters:
        parameter_names = '-'.join(variable_parameters.keys())
        subfolder = os.path.join(output_data_folder, runs_folder, f"{timestamp}--{parameter_names}")
    elif name:
        subfolder = os.path.join(output_data_folder, runs_folder, f"{timestamp}--{name}")
    else:
        subfolder = os.path.join(output_data_folder, runs_folder, f"{timestamp}")

    # Create the subfolder if it doesn't exist
    os.makedirs(subfolder, exist_ok=True)

    return subfolder

def create_heatmap(df, z, color_scale, x_axis_title='x', y_axis_title='y', output_folder=None):
    pass
    # # Calculate the time step intervals
    # time_steps = sorted(df['time_step'].unique())
    # num_time_steps = len(time_steps)
    # no_time_steps_to_plot = 5
    # time_step_interval = max(1, num_time_steps // (no_time_steps_to_plot - 1))
    # time_vals_to_plot = np.floor(np.linspace(1, (no_time_steps_to_plot - 1) * time_step_interval, no_time_steps_to_plot))

    # fig = make_subplots(
    #     rows=1,
    #     cols=no_time_steps_to_plot,
    #     shared_yaxes=True,
    #     subplot_titles=[f'Time Step {time_step}' for time_step in time_steps]
    # )

    # for i, time_step in enumerate(time_vals_to_plot):
    #     # df_at_time_step = df.query("time_step == @time_step and Agent_type == 'Person'")
    #     df_at_time_step = df.query("time_step == @time_step")

    #     heatmap = go.Heatmap(
    #         x=df_at_time_step['x'],
    #         y=df_at_time_step['y'],
    #         z=df_at_time_step[z],
    #         colorscale=color_scale,
    #         colorbar=dict(title=z),
    #         showscale=False
    #     )
    #     fig.add_trace(heatmap, row=1, col=i + 1)

    #     # Label x and y axes for each subplot
    #     fig.update_xaxes(title_text=x_axis_title, row=1, col=i + 1)
    # fig.update_yaxes(title_text=y_axis_title, row=1, col=i + 1)

    # # Add a single color bar to the layout
    # fig.add_trace(go.Heatmap(x=[None], y=[None], z=[[min(df[z]), max(df[z])]], colorscale=color_scale, colorbar=dict(title=z)),
    #               row=1, col=no_time_steps_to_plot)

    # fig.update_layout(
    #     title_text=z,
    #     width=1000,
    #     height=350,
    #     title_x=0.5,   # Center the title
    #     title_y=0.97,  # Adjust the title position
    # )

    # if output_folder:
    #     output_path = os.path.join(output_folder, f'heatmap_{z}.png')
    #     fig.write_image(output_path)
    # else:
    #     fig.show()

def run_experiment(variable_parameters, fixed_parameters, batch_parameters):
    fixed_parameters['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    subfolder = get_subfolder(fixed_parameters['timestamp'], variable_parameters)
    fixed_parameters['subfolder'] = subfolder
    model_parameters = {**fixed_parameters, **variable_parameters}
    with metadata_recorder(model_parameters, batch_parameters, subfolder):
        run_batch_simulation(model_parameters, batch_parameters, subfolder)


# Main execution
if __name__ == '__main__':
    fixed_parameters['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    subfolder = get_subfolder(fixed_parameters['timestamp'], variable_parameters)
    fixed_parameters['subfolder'] = subfolder
    model_parameters = {**fixed_parameters, **variable_parameters}
    with metadata_recorder(model_parameters, batch_parameters, subfolder):
        run_batch_simulation(model_parameters, batch_parameters, subfolder)