import pandas as pd
import argparse
import model.parameters as params

import utils.plotting as plotting
import utils.file_utils as file_utils

num_steps  = 50   
timestamp  = file_utils.generate_timestamp()

parameters = {
    'run_notes': 'Debugging model.',
    'timestamp': timestamp,
    'width':     50,
    'height':    50,
}

variable_parameters = {
    'interventions_on': [True, False],
    # 'distances': [None]
    #   'c': [500, 300, 200], 
    #    'price_of_output': [6.6, 10, 15]
    #   'density': [1200, 600, 100],#[600, 100, 1],
    #    'A': [3000, 2500, 2000],
    #    'alpha': [0.19, 0.18, 0.17],
    #    'beta':  [0.77, 0.75, 0.73],
    #   'gamma': [0.14, 0.12, 0.10],  
    # 'overhead': 0.5,
    # 'mult': 1.2,
    #  'adjN': [0.2, 0.15, 0.1, 0.05, 0.02],
    # 'adjk': 0.10,
    # 'adjn': 0.15,
    # 'adjF': 0.02,
    # 'adjw': [0.09 ,.06, .03], 
    # 'dist': 1, 
    # 'init_F': 100.0,
    # 'init_k': 500.0,
    # 'init_n': 100.0,  
    #   `'wealth_sensitivity': [0.15, 0.1, 0.5],
    #   'cg_tax_per':   [0.5, 0.4, 0.35, .3, .2],# 3, 4, 5], # share 0-1
    #    'cg_tax_invest': [0.5, 0.35, 0.2], #*, # share 0-1
    #   'subsistence_wage': [60000, 40000 30000], # [10000, 30000],
    #    'property_tax_rate': [0.08, 0.04, .02]
    #    'r_prime': [0.12, 0.05, .02], 
    #    'r_investor': [0.1, 0.05, .01]
    # 'r_investor': [0.025] #, 0.05, .1] 
    #    'gamma': [0.001, 0.02, 0.7]
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

def fast_run():
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
        city, agent_out, model_out = run()
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
        city, agent_out, model_out = fast_run()
        if args.plot:
            print('TODO Add plot')
    elif args.fast_batch:
        print('Fast batch')
        df, variable_parameters = fast_batch()
        if args.plot:
            print('Plot fast batch')
            plotting.variables_vs_time(df, variable_parameters)