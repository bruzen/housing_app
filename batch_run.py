import sys
import yaml
import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
from contextlib import contextmanager
from mesa.batchrunner import batch_run
from model.model import City

# Define the variable and fixed parameters
variable_parameters = {
    'density': [1, 100],
    'subsistence_wage': [10000, 30000],
    'gamma': [0.001, 0.02, 0.7]
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
    'data_collection_period': 2,
    'iterations': 1,
    'max_steps': 20
}

# Define the context manager to record metadata
@contextmanager
def metadata_recorder(model_parameters, batch_parameters):
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
def run_batch_simulation():    
    # Run the batch simulations
    results = batch_run(City, model_parameters, **batch_parameters)
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(subfolder, f'batch_results.csv'), index=False)

    # Create a line plot (example: warranted_price vs time_step)
    plt.figure(figsize=(10, 6))
    plt.plot(df['time_step'], df['warranted_price'], label='Warranted Price', marker='o')
    plt.xlabel('Time Step')
    plt.ylabel('Warranted Price')
    plt.title('Warranted Price vs Time Step')
    plt.legend()

    # Create the figures subfolder if it doesn't exist
    figures_folder = os.path.join(subfolder, 'figures')
    os.makedirs(figures_folder, exist_ok=True)

    # Save the plot to the figures subfolder
    plot_path = os.path.join(figures_folder, 'warranted_price_vs_time_step.png')
    plt.savefig(plot_path)

def get_subfolder(timestamp, variable_parameters):
    # Create the subfolder path
    output_data_folder = 'output_data'
    runs_folder = 'batch_runs'
    parameter_names = '-'.join(variable_parameters.keys())
    subfolder = os.path.join(output_data_folder, runs_folder, f"{timestamp}--{parameter_names}")

    # Create the subfolder if it doesn't exist
    os.makedirs(subfolder, exist_ok=True)

    return subfolder

# Main execution
if __name__ == '__main__':
    fixed_parameters['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    subfolder = get_subfolder(fixed_parameters['timestamp'], variable_parameters)
    fixed_parameters['subfolder'] = subfolder
    model_parameters = {**fixed_parameters, **variable_parameters}
    with metadata_recorder(model_parameters, batch_parameters):
        run_batch_simulation()
