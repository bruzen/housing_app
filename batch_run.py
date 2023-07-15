import sys
import yaml
import datetime
import pandas as pd
import os
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
    'subfolder': None,
    'run_notes': 'Debugging model.',
    'width': 50,
    'height': 1,
    'init_city_extent': 10.,  # f CUT OR CHANGE?
    'seed_population': 10,
    'density': 300,
    'subsistence_wage': 40000.,  # psi
    'init_wage_premium_ratio': 0.2,
    'workforce_rural_firm': 100,
    'price_of_output': 1.,  # TODO CUT?
    'alpha_F': 0.18,
    'beta_F': 0.72,  # beta and was lambda, workers_share of aglom surplus
    'beta_city': 1.12,
    'gamma': 0.02,  # FIX value
    'Z': 0.5,  # CUT? Scales new entrants
    'firm_adjustment_parameter': 0.25,
    'wage_adjustment_parameter': 0.5,
    'mortgage_period': 5.0,  # T, in years
    'working_periods': 40,  # in years
    'savings_rate': 0.3,
    'r_prime': 0.05,  # 0.03
    'discount_rate': 0.07, # 1/delta
    'r_margin': 0.01,
    'property_tax_rate': 0.04,  # tau, annual rate, was c
    'housing_services_share': 0.3,  # a
    'maintenance_share': 0.2,  # b
    'max_mortgage_share': 0.9,
    'ability_to_carry_mortgage': 0.28,
    'wealth_sensitivity': 0.1,
}

batch_parameters = {
    'data_collection_period': 2,
    'iterations': 1,
    'max_steps': 3
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
