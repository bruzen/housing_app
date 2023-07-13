import sys
import yaml
import pandas as pd
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
    'max_steps': 30
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
    with open('metadata_batch_run.yaml', 'w') as f:
        yaml.safe_dump(metadata, f)

# Define the function to run the batch simulation
def run_batch_simulation():
    # Run the batch simulations
    results = batch_run(City, variable_parameters, **other_parameters)
    df = pd.DataFrame(results)
    df.to_csv('batch_results.csv', index=False)

# Main execution
if __name__ == '__main__':

    with metadata_recorder(variable_parameters, other_parameters):
        run_batch_simulation()