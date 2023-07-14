import sys
import yaml
import pandas as pd
import os
from contextlib import contextmanager
from mesa.batchrunner import batch_run
from model.model import City

# Define the variable and fixed parameters
variable_parameters = {
    'density': [1, 100],
    'subsistence_wage': [10000, 30000]
}

other_parameters = {
    'data_collection_period': 2,
    'iterations': 1,
    'max_steps': 3
}

# Define the context manager to record metadata
@contextmanager
def metadata_recorder(parameters, other_parameters):
    metadata = {
        'variable_parameters': parameters,
        'other_parameters': other_parameters
    }
    yield metadata
    # Save the metadata to a YAML file
    metadata_path = os.path.join(subfolder, 'batch_metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata, f)

    
# Define the function to run the batch simulation
def run_batch_simulation():
    # Run the batch simulations
    results = batch_run(City, variable_parameters, **other_parameters)
    df = pd.DataFrame(results)
    # df.to_csv('batch_results.csv', index=False)
    df.to_csv(os.path.join(subfolder, 'batch_results.csv'), index=False)

def get_subfolder():
    # Create the subfolder path
    output_data_folder = 'output_data'
    runs_folder = 'batch_runs'
    subfolder = os.path.join(output_data_folder, runs_folder)

    # Create the subfolder if it doesn't exist
    os.makedirs(subfolder, exist_ok=True)

    return subfolder

# Main execution
if __name__ == '__main__':
    subfolder = get_subfolder()
    with metadata_recorder(variable_parameters, other_parameters):
        run_batch_simulation()