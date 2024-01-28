import pandas as pd
import argparse
import model.parameters as params

import utils.plotting as plotting
import utils.file_utils as file_utils

num_steps  = 10
timestamp  = file_utils.generate_timestamp()

parameters = {
    'run_notes': 'Debugging model.',
    'timestamp': timestamp,
    'width':     2,
    'height':    2,
}

variable_parameters = {
    'density': [600, 100, 1],
    # 'subsistence_wage': [10000, 30000],
    # 'gamma': [0.02]   
}

batch_parameters = {
    'data_collection_period': 2,
    'iterations': 1,
    'max_steps': num_steps
}

def run():
    from model.model import City
    city = City(num_steps, **parameters)
    city.run_model()
    agent_out = city.datacollector.get_agent_vars_dataframe()
    model_out = city.datacollector.get_model_vars_dataframe()
    return city, agent_out, model_out

def batch():
    from model.model import City
    from mesa.batchrunner import batch_run
    model_parameters = {**parameters, **variable_parameters} # OR model_parameters = variable_parameters
    results = batch_run(City, model_parameters, **batch_parameters)
    # results = batch_run(Fast, model_parameters, **batch_parameters)
    df = pd.DataFrame(results)

    metadata_filepath = file_utils.get_metadata_filepath(f'metadata-{timestamp}.json')
    metadata = file_utils.record_metadata(filepath = metadata_filepath, timestamp = timestamp, batch_parameters = batch_parameters, variable_parameters = variable_parameters)
    data_filepath = file_utils.get_data_filepath(f'results-batch-{timestamp}.csv')
    df.to_csv(data_filepath, index=False)

    return df, variable_parameters # metadata

def fast():
    from model.model_fast import Fast
    city = Fast(num_steps, **parameters)
    city.run_model()
    agent_out = city.datacollector.get_agent_vars_dataframe()
    model_out = city.datacollector.get_model_vars_dataframe()    
    return city, agent_out, model_out

def fast_batch():
    from model.model_fast import Fast
    from mesa.batchrunner import batch_run
    model_parameters = {**parameters, **variable_parameters} # OR model_parameters = variable_parameters
    results = batch_run(Fast, model_parameters, **batch_parameters)
    # results = batch_run(Fast, model_parameters, **batch_parameters)
    df = pd.DataFrame(results)

    metadata_filepath = file_utils.get_metadata_filepath(f'fast-metadata-{timestamp}.json')
    metadata = file_utils.record_metadata(filepath = metadata_filepath, timestamp = timestamp, batch_parameters = batch_parameters, variable_parameters = variable_parameters)
    data_filepath = file_utils.get_data_filepath(f'fast-results-batch-{timestamp}.csv')
    df.to_csv(data_filepath, index=False)

    return df, variable_parameters # metadata

if __name__ == "__main__":
    # TODO add --all to run all
    # Turn on for timing
    # cProfile.run("agent_out, model_out = run_simulation(num_steps, parameters)", sort='cumulative')
    parser = argparse.ArgumentParser(description="Run the housing market model with configuration to select batch or single run, main model or model_fast, and plotting or no plotting.")

    # Define mutually exclusive group for configuration options with --single as the default
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--run', action='store_true', help='Run a single simulation of the main model')
    group.add_argument('--batch', action='store_true', help='Run the main model in batch mode')
    group.add_argument('--fast_run', action='store_true', help='Run a single simulation of model_fast')
    group.add_argument('--fast_batch', action='store_true', help='Run model_fast in batch mode')

    # Add an option for plotting
    parser.add_argument('--plot', action='store_true',default=True, help='Enable plotting (default: disabled)')

    args = parser.parse_args()

    # Convert argparse Namespace to dictionary
    config = vars(args)

    # Set default value
    if not any(vars(args).get(key, False) for key in ['run', 'batch', 'fast_run', 'fast_batch']):
        args.fast_batch = True

    # Import and call the appropriate module based on the selected configuration
    if args.run:
        print('Run')
        city, agent_out, model_out = main()
        if args.plot:
            print('TODO Add plot')
    elif args.batch:
        print('Batch')
        df, variable_parameters = batch()
        if args.plot:
            print('Plot batch')
            plotting.variables_vs_time(df, variable_parameters)
    elif args.fast_run:
        print('Fast run')
        if args.plot:
            print('TODO Add plot')
    elif args.fast_batch:
        print('Fast batch')
        df, variable_parameters = fast_batch()
        if args.plot:
            print('Plot fast batch')
            plotting.variables_vs_time(df, variable_parameters)