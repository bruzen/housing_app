import argparse
import model.parameters as params

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the housing market model with configuration to select batch or single run, main model or model_fast, and plotting or no plotting.")

    # Define mutually exclusive group for configuration options with --single as the default
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--run', action='store_true', help='Run a single simulation of the main model')
    group.add_argument('--batch', action='store_true', help='Run the main model in batch mode')
    group.add_argument('--fast_run', action='store_true', help='Run a single simulation of model_fast')
    group.add_argument('--fast_batch', action='store_true', help='Run model_fast in batch mode')

    # Add an option for plotting
    parser.add_argument('--plot', action='store_true', help='Enable plotting (default: disabled)')

    args = parser.parse_args()

    # Convert argparse Namespace to dictionary
    config = vars(args)

    # If no option is specified, set --run to True
    if not any(vars(args).values()):
        args.run = True

# Import and call the appropriate module based on the selected configuration
if args.run:
    from model.model import City
    num_steps  = params.steps
    city = City(num_steps)
    city.run_model()

    # agent_out = city.datacollector.get_agent_vars_dataframe()
    model_out = city.datacollector.get_model_vars_dataframe()

    model_out.to_csv('___test_model_out.csv', index=False)

    print('run - test')
    if args.plot:
        print('plot')
elif args.batch:
    # from model.model import City
    # from mesa.batchrunner import batch_run
    print('batch run - test')
    if args.plot:
        print('plot')
elif args.fast_run:
    # from model.model_fast import Fast
    print('fast run - test')
    if args.plot:
        print('plot')
elif args.fast_batch:
    # from model.model_fast import Fast
    # from mesa.batchrunner import batch_run
    print('batch fast run - test')
    if args.plot:
        print('plot')