import logging
import os
import yaml
# import functools
import datetime
import random
import string
from typing import Dict, List
from contextlib import contextmanager
# import subprocess
# import math
import numpy as np
import pandas as pd
from scipy.spatial import distance

from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error
# from scikit-learn import linear_model
# import statsmodels.api as sm
# from pysal.model import spreg

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from model.agents import Land, Person, Firm, Investor, Bank, Realtor
from model.schedule import RandomActivationByBreed

logging.basicConfig(filename='logfile.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

logging.getLogger('matplotlib').setLevel(logging.ERROR) 

# def capture_rents(model):
#     """Current rents for each location in the grid."""
#     rent_grid = []
#     for row in model.grid.grid:   
#         new_row = []
#         for cell in row:
#             cell_rent = -1.0
#             for item in cell:
#                 try:
#                     if (item.residence):
#                         cell_rent = item.rent
#                 except:
#                     pass
#             new_row.append(cell_rent)
#         rent_grid.append(new_row)
#     return rent_grid

class City(Model):
    @property
    def city_extent_calc(self):
        # Compute urban boundary where it is not worthwhile to work
        return self.firm.wage_premium /  self.transport_cost_per_dist

    @property
    def r_target(self):
        return self.r_prime + self.r_margin

    def __init__(self, num_steps=10, **parameters):
        super().__init__()

        # Default parameter values
        default_parameters = {
            'run_notes': 'Debugging model.',
            'subfolder': None,
            'width': 50,
            'height': 1,
            'center_city': False,     # Flag for city center in center if True, or bottom corner if False
            'random_init_age': False, # Flag for randomizing initial age. If False, all workers begin at age 0
            'init_city_extent': 10.,  # f CUT OR CHANGE?
            'seed_population': 400,
            'density': 300,
            'subsistence_wage': 40000.,
            'init_wage_premium_ratio': 1.2,
            # 'workforce_rural_firm': 100,
            # 'alpha_F': 0.18,
            # 'beta_F': 0.72,  # beta and was lambda, workers_share of aglom surplus
            # 'beta_city': 1.12,
            # 'gamma': 0.02,  # FIX value
            # 'Z': 0.5,  # CUT? Scales new entrants
            'alpha': 0.18,
            'beta':  0.73,
            'gamma': 0.11,
            'price_of_output': 4., 
            'r_prime': 0.05,  # 0.03
            # 'firm_adjustment_parameter': 0.25,
            # 'wage_adjustment_parameter': 0.5,
            'mortgage_period': 5.0,  # T, in years
            'working_periods': 40,  # in years
            'savings_rate': 0.3,
            'discount_rate': 0.07, # 1/delta
            'r_margin': 0.01,
            'property_tax_rate': 0.04,  # tau, annual rate, was c
            'housing_services_share': 0.3,  # a
            'maintenance_share': 0.2,  # b
            'max_mortgage_share': 0.9,
            'ability_to_carry_mortgage': 0.28,
            'wealth_sensitivity': 0.1,
            'A': 500,
            'overhead': 2,
            'mult': 1.2,
            'c': 200.0,
            'adjN': 0.15,   #### Sept 28 raise adj factors from 15, 15, 25, 15, 15
            'adjk': 0.15,
            'adjn': 0.25,
            'adjF': 0.15,
            'adjw': 0.15,
            'dist': 1.0,
            'init_agglomeration_population': 0.0,
            'init_F': 100.0,
            'init_k': 100.0,
            'init_n': 100.0,
        }

        # Merge default parameters with provided parameters
        if parameters is not None:
            self.params = {**default_parameters, **parameters}
        else:
            self.params = default_parameters

        # Model
        self.model_name        = 'Housing Market'
        self.model_version     = '0.0.1'
        self.model_description = 'Agent-based housing market model with rent and urban aglomeration.'
        self.num_steps = num_steps        
        self.time_step = 1.
        self.height = self.params['height']
        self.width  = self.params['width']
        # If self.center_city is True, it places the city in the center; otherwise, it places it in the bottom corner.
        self.center_city = self.params['center_city'] # put city in the bottom corner TODO check flag's logic
        if self.center_city:
            self.center    = (width//2, height//2)
        else:
            self.center    = (0, 0)
        self.grid = MultiGrid(self.params['width'], self.params['height'], torus=False)
        self.schedule = RandomActivationByBreed(self)
        self.transport_cost_per_dist = self.params['init_wage_premium_ratio'] * self.params['subsistence_wage'] / self.params['init_city_extent'] # c

        # People
        self.working_periods  = self.params['working_periods']
        self.savings_per_step = self.params['subsistence_wage'] * self.params['savings_rate']

        # Housing market model
        self.mortgage_period        = self.params['mortgage_period']
        self.housing_services_share = self.params['housing_services_share'] # a
        self.maintenance_share      = self.params['maintenance_share'] # b
        self.r_prime  = self.params['r_prime']
        self.r_margin = self.params['r_margin']
        self.delta    = 1/self.params['discount_rate'] # TODO divide by zero error checking
        self.max_mortgage_share        = self.params['max_mortgage_share']
        self.ability_to_carry_mortgage = self.params['ability_to_carry_mortgage']
        self.wealth_sensitivity        = self.params['wealth_sensitivity']
        self.p_dot                     = 0.0 # Price adjustment rate. TODO fix here? rename?
        self.warranted_price_model     = None
        self.realized_price_model      = None

        # Add workforce manager to track workers, newcomers, and retiring agents
        self.workforce = Workforce()
        self.removed_agents = 0

        # Add bank, firm, investor, and realtor
        self.unique_id       = 1        
        self.bank            = Bank(self.unique_id, self, self.center, self.r_prime)
        self.grid.place_agent(self.bank, self.center)
        self.schedule.add(self.bank)
        
        self.unique_id      += 1
        self.firm            = Firm(self.unique_id, self, self.center, 
                                    self.params['subsistence_wage'],
                                    self.params['init_wage_premium_ratio'],
                                    self.params['alpha'], self.params['beta'], self.params['gamma'],
                                    self.params['price_of_output'], self.params['r_prime'],
                                    # self.params['wage_adjustment_parameter'],
                                    # self.params['firm_adjustment_parameter'],
                                    self.params['seed_population'],
                                    self.params['density'],
                                    A=self.params['A'],
                                    overhead=self.params['overhead'],
                                    mult=self.params['mult'],
                                    c=self.params['c'],
                                    adjN=self.params['adjN'],
                                    adjk=self.params['adjk'],
                                    adjn=self.params['adjn'],
                                    adjF=self.params['adjF'],
                                    adjw=self.params['adjw'],
                                    dist=self.params['dist'],
                                    init_agglomeration_population=self.params['init_agglomeration_population'],
                                    init_F=self.params['init_F'],
                                    init_k=self.params['init_k'],
                                    init_n=self.params['init_n']
                                    )
        self.grid.place_agent(self.firm, self.center)
        self.schedule.add(self.firm)

        self.unique_id      += 1
        self.investor        = Investor(self.unique_id, self, self.center)
        self.grid.place_agent(self.investor, self.center)
        self.schedule.add(self.investor)

        self.unique_id      += 1
        self.realtor         = Realtor(self.unique_id, self, self.center)
        self.grid.place_agent(self.realtor, self.center)
        self.schedule.add(self.realtor)

        # Add land and people to each cell
        self.unique_id      += 1
        for cell in self.grid.coord_iter():
            pos              = (cell[1], cell[2])

            land             = Land(self.unique_id, self, pos, 
                                    self.params['property_tax_rate'])
            self.grid.place_agent(land, pos)
            self.schedule.add(land)

            self.unique_id      += 1
            # TODO maybe control flow for this with a flag passed in
            if self.params['random_init_age']:
                init_working_period  = self.random.randint(0, self.working_periods - 1) # TODO randomize working period
            else:
                init_working_period  = 0
            savings = init_working_period * self.savings_per_step 
            # TODO check boundaries - at working period 0, no savings
            person  = Person(self.unique_id, self, pos,
                                init_working_period = init_working_period,
                                savings             = savings,
                                residence_owned     = land)
            self.grid.place_agent(person, pos)
            self.schedule.add(person)

            self.unique_id  += 1

        self.setup_data_collection()

    def step(self):
        """ The model step function runs in each time step when the model
        is executed. It calls the agent functions, then records results
        for analysis.
        """

        self.time_step += 1

        logger.info(f'\n \n \n Step {self.schedule.steps}. \n')
        self.step_price_data.clear()

        # Land records locational rents and calculates price forecast
        self.schedule.step_breed(Land)
        new_df = pd.DataFrame(self.step_price_data)
        self.warranted_price_data = pd.concat([self.warranted_price_data, new_df], 
                                          ignore_index=True)

        self.p_dot       = self.get_p_dot()

        # Firms update wages
        self.schedule.step_breed(Firm)
    
        # People work, retire, and list homes to sell
        self.schedule.step_breed(Person)

        for i in self.workforce.retiring:
            # Add agents to replace retiring workers
            person = self.create_newcomer()
            person.bid()

        # Investors bid on properties
        self.schedule.step_breed(Investor, step_name='bid')

        # Realtors sell homes
        self.schedule.step_breed(Realtor, step_name='sell_homes')

        # Realtors rent properties
        self.schedule.step_breed(Realtor, step_name='rent_homes')

        # Advance model time
        self.schedule.step_time()

        self.record_step_data()

    def run_model(self):
        for t in range(self.num_steps):
            self.step()

        self.record_run_data_to_file()

    def setup_data_collection (self):

        # Variables for data collection
        self.rent_production = 0.
        self.rent_amenity    = 0.
        self.market_rent     = 0.
        self.net_rent        = 0.
        self.potential_dissipated_rent  = 0.
        self.dissipated_rent = 0.
        self.available_rent  = 0.
        self.rent_captured_by_finance  = 0.
        self.share_captured_by_finance = 0.
        self.urban_surplus   = 0.

        # Setup data collection
        if 'timestamp' in self.params and self.params['timestamp'] is not None:
            timestamp = self.params['timestamp']
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.timestamp = timestamp

        if 'subfolder' in self.params and self.params['subfolder'] is not None:
            subfolder = self.params['subfolder']
        else:
            subfolder = self.get_subfolder()
        self.subfolder = subfolder    

        self.run_id    = self.get_run_id(self.model_name, self.timestamp, self.model_version)

        # TODO FIX TO IN WORKERS-- RECORD WORKERS, NEWCOMERS, RETIRING
        # Define what data the model will collect in each time step
        model_reporters      = {
#           "rents":                     capture_rents,
            "workers":                   lambda m: m.firm.N,
            "MPL":                       lambda m: m.firm.MPL,
            #"MPK":                       lambda m: m.firm.MPK,
            "time_step":                 lambda m: m.time_step,
            "companies":                 lambda m: m.schedule.get_breed_count(Firm),
            "city_extent_calc":          lambda m: m.city_extent_calc,
            "people":                    lambda m: m.schedule.get_breed_count(Person),
            "worker_agents":             lambda m: m.workforce.get_agent_count(m.workforce.workers),
            "market_rent":               lambda m: m.market_rent,
            "net_rent":                  lambda m: m.net_rent,
            "potential_dissipated_rent": lambda m: m.potential_dissipated_rent,
            "dissipated_rent":           lambda m: m.dissipated_rent,
            "available_rent":            lambda m: m.available_rent,
            "rent_captured_by_finance":  lambda m: m.rent_captured_by_finance,
            "share_captured_by_finance": lambda m: m.share_captured_by_finance,
            "urban_surplus":             lambda m: m.urban_surplus,
            "p_dot":                     lambda m: m.p_dot,
            "removed_agents":            lambda m: m.removed_agents,
            "n":                         lambda m: m.firm.n,
            "y":                         lambda m: m.firm.y,
            #"n_target":                  lambda m: m.firm.n_target,
            # "y_target":                  lambda m: m.firm.y_target,
            # "k_target":                  lambda m: m.firm.k_target, 
            # # "self.firm_adjustment_parameter": lambda m: m.firm.firm_adjustment_parameter,
            "F_target":                  lambda m: m.firm.F_target,
            # "F_next":                    lambda m: m.firm.F_next,
            # "N_target_total":            lambda m: m.firm.N_target_total,
            # "F_next_total":              lambda m: m.firm.F_next_total,
            "F":                         lambda m: m.firm.F,
            "k":                         lambda m: m.firm.k,
            "N":                         lambda m: m.firm.N,
            "agglomeration_population":  lambda m: m.firm.agglomeration_population,
            "Y":                         lambda m: m.firm.Y,
            "wage_premium":              lambda m: m.firm.wage_premium,
            "subsistence_wage":          lambda m: m.firm.subsistence_wage,
            "wage":                      lambda m: m.firm.wage,

            # "A_F":                     lambda m: m.firm.A_F,

            # "price_model_coefficients":  lambda m: m.price_model.coef,
            # "price_model_intercept":     lambda m: m.price_model.intercept,
            # "workers":        lambda m: len(
            #     [a for a in self.schedule.agents_by_breed[Person].values()
            #              if a.is_working == 1]
            # )
        }

        agent_reporters      = {
            "time_step":         lambda a: a.model.time_step,
            "agent_class":       lambda a: type(a),
            "agent_type":        lambda a: type(a).__name__,
            "id":                lambda a: a.unique_id,
            "x":                 lambda a: a.pos[0],
            "y":                 lambda a: a.pos[1],
            "distance_from_center": lambda a: getattr(a, "distance_from_center", None) if isinstance(a, Land) else None,
            # "wage":               lambda a: getattr(a, "wage", None) if isinstance(a, Land) else None,
            "is_working":           lambda a: None if not isinstance(a, Person) else 1 if a.unique_id in a.workforce.workers else 0,
            # "is_working":         lambda a: getattr(a, "is_working", None),
            "working_period":    lambda a: getattr(a, "working_period", None),
            # "property_tax_rate":  lambda a: getattr(a, "property_tax_rate", None),
            "net_rent":          lambda a: getattr(a, "net_rent", None) if isinstance(a, Land) else None,
            "warranted_rent":    lambda a: getattr(a, "warranted_rent", None) if isinstance(a, Land) else None,
            "warranted_price":   lambda a: getattr(a, "warranted_price", None) if isinstance(a, Land) else None,
            "person_vs_investor_owner":   lambda a: getattr(a, "person_vs_investor_owner", None) if isinstance(a, Land) else None,
            "realized_price":   lambda a: getattr(a, "realized_price", None) if isinstance(a, Land) else None,
            "owner_type":   lambda a: getattr(a, "owner_type", None) if isinstance(a, Land) else None,
        }
        # TODO replace with actual values for "person_vs_investor_owner" and "realized_price":

        self.datacollector  = DataCollector(model_reporters = model_reporters,
                                            agent_reporters = agent_reporters)

        # Price data for forecasting
        self.warranted_price_data = pd.DataFrame(
             columns=['land_id', 'warranted_price', 'time_step', 'transport_cost', 'wage'])   
        self.step_price_data = []
        self.realized_price_data  = pd.DataFrame(
             columns=['land_id', 'realized_price', 'time_step', 'transport_cost', 'wage']) 

        # Create the 'output_data' subfolder if it doesn't exist
        if not os.path.exists(self.subfolder):
            os.makedirs(self.subfolder)

        agent_filename         = self.run_id + '_agent' + '.csv'
        model_filename         = self.run_id + '_model' + '.csv'
        self.agent_file_path   = os.path.join(self.subfolder, agent_filename)
        self.model_file_path   = os.path.join(self.subfolder, model_filename)
        self.metadata_file_path = os.path.join(self.subfolder, 'run_metadata.yaml')

        metadata = {
            'model_description':     self.model_description,
            'num_steps':             self.num_steps,
            'simulation_parameters': self.params
        }

        self.record_metadata(metadata, self.metadata_file_path)

    def record_metadata(self, metadata, metadata_file_path):
        """Append metadata for each experiment to a metadata file."""

        # Check if the file exists
        file_exists = os.path.isfile(metadata_file_path)

        # If the file exists, load the existing metadata; otherwise, create an empty dictionary
        if file_exists:
            with open(metadata_file_path, 'r') as file:
                existing_metadata = yaml.safe_load(file)
        else:
            existing_metadata = {}

        # Append the metadata for the current experiment to the existing metadata dictionary
        existing_metadata[self.run_id] = metadata

        # Write the updated metadata back to the file
        with open(metadata_file_path, 'w') as file:
            yaml.safe_dump(existing_metadata, file)

    def record_step_data(self):
        # Calculations for data collection
        # TODO: Check only one worker per house and that all workers have a residence
        self.rent_production = sum(
            agent.model.firm.wage_premium for agent in self.schedule.agents_by_breed[Person].values() 
            if agent.unique_id in agent.workforce.workers
        )

        # TODO  Do we only count amenity for workers, or those in the urban boundary?
        self.rent_amenity    = sum(
            agent.amenity for agent in self.schedule.agents_by_breed[Person].values() 
            if agent.unique_id in agent.workforce.workers
        )

        self.market_rent = sum(agent.market_rent    for agent in self.schedule.agents_by_breed[Land].values()
                               if agent.resident and agent.resident.unique_id in agent.resident.workforce.workers)
        self.net_rent    = sum(agent.net_rent       for agent in self.schedule.agents_by_breed[Land].values()
                               if agent.resident and agent.resident.unique_id in agent.resident.workforce.workers)
        self.potential_dissipated_rent = sum(agent.transport_cost for agent in self.schedule.agents_by_breed[Land].values())
        self.dissipated_rent = sum(
            agent.transport_cost for agent in self.schedule.agents_by_breed[Land].values() 
            if agent.resident and agent.resident.unique_id in agent.resident.workforce.workers
        )
        self.available_rent  = self.rent_production + self.rent_amenity - self.dissipated_rent # w - cd + A - total_dissipated # total-captured
        self.rent_captured_by_finance  = 0 # TODO implement. make a marker for agents in the city
        self.share_captured_by_finance = 0 # TODO implement.
        self.urban_surplus   = 0 # TODO implement

        # Retrieve data
        self.datacollector.collect(self)

    def record_run_data_to_file(self):
        model_out = self.datacollector.get_model_vars_dataframe()
        agent_out = self.datacollector.get_agent_vars_dataframe()

        # Save agent data
        if agent_out is not None:
            try:
                agent_out.to_csv(self.agent_file_path, index=False)
            except Exception as e:
                logging.error("Error saving agent data: %s", str(e))

        # Save model data
        if model_out is not None:
            try:
                model_out.to_csv(self.model_file_path, index=False)
            except Exception as e:
                logging.error("Error saving model data: %s", str(e))

    def get_run_id(self, model_name, timestamp, model_version):
        # Adapt the model name to lowercase and replace spaces with underscores
        formatted_model_name = model_name.lower().replace(" ", "_")
        unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

        # Create the run_id
        return f"{formatted_model_name}_{timestamp}_{unique_id}_v{model_version.replace('.', '_')}"

    def get_subfolder(self):
        # Create the subfolder path
        output_data_folder = "output_data"
        runs_folder = "runs"
        subfolder = os.path.join(output_data_folder, runs_folder)
        
        # Create the subfolder if it doesn't exist
        os.makedirs(subfolder, exist_ok=True)
        
        return subfolder

    def create_newcomer(self):
        """Create newcomer at the center with no residence or property."""
        self.unique_id  += 1
        person           = Person(self.unique_id, self, self.center, 
                                  residence_owned = None)
        self.grid.place_agent(person, self.center)
        self.schedule.add(person)
        self.workforce.add(person, self.workforce.newcomers)
        return person

    def get_distance_to_center(self, pos):
        return distance.euclidean(pos, self.center)

    # If there were more data, we might use k-folds
    # def get_price_model(self):
    #     x = self.warranted_price_data[['time_step', 'transport_cost', 'wage']]

    #     # Dependent variable
    #     y = self.warranted_price_data['warranted_price']

    #     # Define the number of folds for cross-validation
    #     num_folds = 5

    #     # Initialize the k-fold cross-validator
    #     kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    #     # Initialize lists to store the coefficients for each fold
    #     coef_list = []

    #     # Perform k-fold cross-validation
    #     for train_index, test_index in kf.split(x):
    #         x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    #         y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #         # Create and train the Linear Regression model
    #         regression_model = LinearRegression()
    #         regression_model.fit(x_train, y_train)

    #         # Store the coefficients for this fold
    #         coef_list.append(regression_model.coef_)

    #     # Calculate the average coefficients across all folds
    #     # average_coef = np.mean(coef_list, axis=0)
    #     self.price_model_coef = np.mean(coef_list, axis=0)
    #     self.price_model_intercept  = np.mean(intercept_list)  # Calculate average intercept

    #     return self.price_model_coef

    # def get_p_dot(self, transport_cost):
    #     """Rate of growth for property price"""
    #     # Make p_dot zero for the first 10 steps or so.
    #     if self.time_step < 5:
    #         p_dot = 0
    #     else: 
    #         # p_dot = np.dot(self.price_model, [self.time_step, transport_cost, self.firm.wage])
    #         coef_time_step, coef_transport_cost, coef_wage = self.price_model
    #         p_dot = (
    #             coef_time_step * self.time_step +
    #             coef_transport_cost * transport_cost +
    #             coef_wage * self.firm.wage
    #         )
    #     return p_dot

    # # Alternative model, using statsmodels
    # x = sm.add_constant(x) # adding a constant
    # model = sm.OLS(y, x).fit()
    # predictions = model.predict(x)

    def get_warranted_price_model(self):
        x_w = self.warranted_price_data[['time_step','transport_cost','wage']] # Independent variables
        warranted_price       = self.warranted_price_data['warranted_price']   # Dependent variable
        warranted_price_model = LinearRegression()
        warranted_price_model.fit(x_w, warranted_price)
        return warranted_price_model

    def get_realized_price_model(self):
        x_r = self.realized_price_data[['time_step','transport_cost','wage']] # Independent variables
        realized_price       = self.realized_price_data['realized_price']     # Dependent variable
        realized_price_model = LinearRegression()
        realized_price_model.fit(x_r, realized_price)
        return realized_price_model

    def get_p_dot(self):
        time_zero = 6
        if self.time_step < time_zero:
            p_dot = 0
        else:
            # Predict rate of change using the warranted price model
            # logger.debug(f'Len warranted_price_data {len(self.warranted_price_data)}')
            warranted_price_model = self.get_warranted_price_model()
            if warranted_price_model is None:
                # Handle the case where the model is not created or trained
                logger.error("Error: Warranted price model is not initialized.")
                p_dot_warranted_price = 0
            else:
                p_dot_warranted_price = warranted_price_model.coef_[0]

            # Predict rate of change using the realized price model        
            if len(self.realized_price_data) > 10:
                # logger.debug(f'Len realized_price_data {len(self.realized_price_data)}')
                realized_price_model = self.get_realized_price_model()
                if realized_price_model is None:
                    # Handle the case where the model is not created or trained
                    logger.error("Error: Realized price model is not initialized.")
                    p_dot_realized_price = 0
                else:
                    p_dot_realized_price = realized_price_model.coef_[0]
            else:
                p_dot_realized_price = 0

            # The influence of the weighted model prediction increases over time
            time_threshold = time_zero * 2
            w_w = min(1.0, self.time_step / time_threshold)

            # The influence of p_dot_realized_price on p_dot increases as the quantity of realized price data grows. 
            data_threshold = 100
            w_r = min(len(self.realized_price_data) / data_threshold, 1.0)

            # Alternative p_dot calculations
            # p_dot = .001 
            # p_dot = p * p_dot_warranted_price + (1 - p) * p_dot_realized_price # Old without, 0 term
            # TODO consider sigmoid transitions

            # Linearly interpolate between 0 and the weighted rate of change
            p_dot = w_w * (w_r * p_dot_warranted_price + (1 - w_r) * p_dot_realized_price)
            
        return p_dot

# TODO MAKE A WEGHTED MODE FOR THE TWO VARIABLES
# Assume you have the trained models regr1 and regr2
# Assume you have data for new_data1 and new_data2 for both models
# p is the weight parameter (a value between 0 and 1)



# Blend the predictions using the weight parameter p


class Workforce:
    """Manages a dictionary of working agents."""

    def __init__(self):
        self.workers:   Dict[int, Person] = {}
        self.retiring:  Dict[int, Person] = {}
        self.newcomers: Dict[int, Person] = {}

    def add(self, agent: Person, agent_dict: dict) -> None:
        if agent.unique_id not in agent_dict:
            agent_dict[agent.unique_id] = agent

    def remove(self, agent: Person, agents_dict: Dict[int, Person]) -> None:
        if agent.unique_id in agents_dict:
            del agents_dict[agent.unique_id]

    # def add(self, agent: Person, agent_dict: dict) -> None:
    #     if agent.unique_id in agent_dict:
    #         logging.warning(f"Person with unique id {agent.unique_id!r} is already in the manager.")
    #         return  # If agent is already present, simply return without adding it again.

    #     agent_dict[agent.unique_id] = agent

    # def remove(self, agent: Person, agents_dict: Dict[int, Person]) -> None:
    #     if agent.unique_id not in agents_dict:
    #         logging.warning(f"Person with unique id {agent.unique_id!r} not found in the manager")
    #         return  # If agent is not found, simply return without attempting to remove.

    #     del agents_dict[agent.unique_id]

    def remove_from_all(self, agent: Person) -> None:
        # Remove the agent from each dictionary using the `remove` method.
        self.remove(agent, self.workers)
        self.remove(agent, self.retiring)
        self.remove(agent, self.newcomers)

    def get_agent_count(self, agents_dict: Dict[int, Person]) -> int:
        """Returns the current number of agents in the dictionary."""
        return len(agents_dict)

    @property
    def agents(self, agents_dict: Dict[int, Person]) -> List[Person]:
        """Returns a list of all working agents."""
        return list(agents_dict.values())

    def do_each(self, method, agents_dict: Dict[int, Person], shuffle_agents: bool = False) -> None:
        """Execute a method for each working agent.

        Args:
            method: The name of the method to be executed.
            shuffle_agents: Whether to shuffle the order of the agents.

        """
        agent_keys = list(agents_dict.keys())
        if shuffle_agents:
            random.shuffle(agent_keys)

        for agent_key in agent_keys:
            agent = agents_dict[agent_key]
            getattr(agent, method)()