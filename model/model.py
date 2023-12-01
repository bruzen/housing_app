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
            'width': 15,
            'height':15,

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
            'adjk': 0.05,
            'adjn': 0.25,
            'adjF': 0.15,
            'adjw': 0.15, 
            'dist': 1,
            'init_F': 100.0,
            'init_k': 100.0,
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

        # Merge default parameters with provided parameters
        if parameters is not None:
            self.params = {**default_parameters, **parameters}
        else:
            self.params = default_parameters

        # Model
        self.model_name        = 'Housing Market'
        self.model_version     = '0.0.1'
        self.model_description = 'Agent-based housing market model with rent and urban agglomeration.'
        self.num_steps = num_steps        
        self.time_step = 0.
        self.height = self.params['height']
        self.width  = self.params['width']

        # Initialize counters
        self.urban_investor_owners_count = 0
        self.urban_resident_owners_count = 0
        self.urban_other_owners_count    = 0

        # # Set the random seed for reproducibility
        # self.random_seed = 42
        # self.random.seed(self.random_seed)

        # If self.center_city is True, it places the city in the center; otherwise, it places it in the bottom corner.
        self.center_city   = self.params['center_city'] # put city in the bottom corner TODO check flag's logic
        if self.center_city:
            self.center    = (width//2, height//2)
        else:
            self.center    = (0, 0)
        self.grid = MultiGrid(self.params['width'], self.params['height'], torus=False)
        self.schedule = RandomActivationByBreed(self)
        self.transport_cost_per_dist = self.params['init_wage_premium_ratio'] * self.params['subsistence_wage'] / self.params['init_city_extent'] # c

        # People
        # If demographics_on, there is a housing market when agents retire # self.demographics_on = self.params['demographics_on']
        if self.params['demographics_on']:
            self.working_periods  = self.params['working_periods']
            logger.debug(f'Demographics on, working periods {self.working_periods}, 2x time steps {self.num_steps}') #, params working periods {self.params['working_periods']}')
        else:
            self.working_periods = 10 * self.num_steps
            logger.debug(f'Demographics off, working periods {self.working_periods}')
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
        # self.warranted_price_model     = None
        # self.realized_price_model      = None

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
                init_working_period  = self.random.randint(0, self.params['working_periods'] - 1) # TODO randomize working period
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
        # self.record_step_data()

    def step(self):
        """ The model step function runs in each time step when the model
        is executed. It calls the agent functions, then records results
        for analysis.
        """

        self.time_step += 1

        # Reset counters
        self.urban_investor_owners_count = 0
        self.urban_resident_owners_count = 0
        self.urban_other_owners_count    = 0

        logger.info(f'\n \n \n Step {self.schedule.steps}. \n')
        self.step_price_data.clear()

        # Firms update wages
        self.schedule.step_breed(Firm)

        # Land records locational rents and calculates price forecast
        self.schedule.step_breed(Land)
        new_df = pd.DataFrame(self.step_price_data)
        self.warranted_price_data = pd.concat([self.warranted_price_data, new_df], 
                                          ignore_index=True)
    
        # People work, retire, and list homes to sell
        self.schedule.step_breed(Person)

        for i in self.workforce.retiring_urban_owner:
            # Add agents to replace retiring_urban_owner workers
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

        # Do checks TEMP
        for cell in self.grid.coord_iter():
            pos                 = (cell[1], cell[2])
            cell_contents       = self.grid.get_cell_list_contents(pos)
            persons_at_position = [agent for agent in cell_contents if isinstance(agent, Person)]

            if len(persons_at_position) > 1:
                num_not_in_newcomers = [agent for agent in persons_at_position if agent not in self.workforce.newcomers]
                if len(num_not_in_newcomers) > 1:
                    # Extra non-newcomers
                    unique_ids = [str(agent.unique_id) for agent in num_not_in_newcomers]
                    comma_separated_unique_ids = ', '.join(unique_ids)
                    logging.warning(f'More than one Person agent in self.workforce.newcomers at location {pos}, agents are {comma_separated_unique_ids}')
                elif len(num_not_in_newcomers == 0):
                    # Only newcomers    
                    logging.warning(f'Only newcomers at location {pos}, agents are {comma_separated_unique_ids}')
                # else:
                #     # Newcomers
                #     unique_ids = [str(agent.unique_id) for agent in persons_at_position]
                #     comma_separated_unique_ids = ', '.join(unique_ids)
                #     logging.debug(f'More than one Person agent at location {pos}, agents are {comma_separated_unique_ids}')
                # TODO Could check that non-newcomer is resident etc.

            elif len(persons_at_position) == 0:
                # There are no people
                logging.warning(f'No Person agents at location {pos}')

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
        self.potential_dissipated_rent = 0.
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

        # Define what data the model will collect in each time step
        model_reporters      = {
            "workers":                   lambda m: m.firm.N,
            "MPL":                       lambda m: m.firm.MPL,
            "time_step":                 lambda m: m.time_step,
            "companies":                 lambda m: m.schedule.get_breed_count(Firm),
            "city_extent_calc":          lambda m: m.city_extent_calc,
            "people":                    lambda m: m.schedule.get_breed_count(Person),
            "market_rent":               lambda m: m.market_rent,
            "net_rent":                  lambda m: m.net_rent,
            "potential_dissipated_rent": lambda m: m.potential_dissipated_rent,
            "dissipated_rent":           lambda m: m.dissipated_rent,
            "available_rent":            lambda m: m.available_rent,
            "rent_captured_by_finance":  lambda m: m.rent_captured_by_finance,
            "share_captured_by_finance": lambda m: m.share_captured_by_finance,
            "urban_surplus":             lambda m: m.urban_surplus,
            "removed_agents":            lambda m: m.removed_agents,
            "n":                         lambda m: m.firm.n,
            "y":                         lambda m: m.firm.y,
            "F_target":                  lambda m: m.firm.F_target,
            "F":                         lambda m: m.firm.F,
            "k":                         lambda m: m.firm.k,
            "N":                         lambda m: m.firm.N,
            # "agglomeration_population":  lambda m: m.firm.agglomeration_population, # TODO delete
            "Y":                         lambda m: m.firm.Y,
            "wage_premium":              lambda m: m.firm.wage_premium,
            "subsistence_wage":          lambda m: m.firm.subsistence_wage,
            "wage":                      lambda m: m.firm.wage,
            # "worker_agents":           lambda m: m.workforce.get_agent_count(m.workforce.workers),
            "worker_agents":             lambda m: len(m.workforce.workers),
            "newcomer_agents":           lambda m: len(m.workforce.newcomers),
            "retiring_urban_owner":      lambda m: len(m.workforce.retiring_urban_owner),
            "urban_resident_owners":     lambda m: m.urban_resident_owners_count,
            "urban_investor_owners":     lambda m: m.urban_investor_owners_count,
            "urban_other_owners":        lambda m: m.urban_other_owners_count,
            "investor_ownership_share":  lambda m: m.urban_investor_owners_count / (m.urban_resident_owners_count + m.urban_investor_owners_count) if (m.urban_resident_owners_count + m.urban_investor_owners_count) != 0 else 1,
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
            "is_working":        lambda a: None if not isinstance(a, Person) else 1 if a.unique_id in a.workforce.workers else 0,  # TODO does this need to be in model? e.g. a.model.workforce
            "is_working_check":  lambda a: None if not isinstance(a, Person) else a.is_working_check,
            "working_period":    lambda a: getattr(a, "working_period", None)  if isinstance(a, Person)       else None,
            "wage_delta":        lambda a: getattr(a, "wage_delta", None)      if isinstance(a, Firm)         else None,
            "p_dot":             lambda a: getattr(a, "p_dot", None)           if isinstance(a, Land)         else None,
            "net_rent":          lambda a: getattr(a, "net_rent", None)        if isinstance(a, Land)         else None,
            "warranted_rent":    lambda a: getattr(a, "warranted_rent", None)  if isinstance(a, Land)         else None,
            "warranted_price":   lambda a: getattr(a, "warranted_price", None) if isinstance(a, Land)         else None,
            "realized_price":    lambda a: getattr(a, "realized_price", None)  if isinstance(a, Land)         else None,
            "sold_this_step":    lambda a: getattr(a, "sold_this_step", None)  if isinstance(a, Land)         else None,
            "ownership_type":    lambda a: getattr(a, "ownership_type", None)  if isinstance(a, Land)         else None,
            "distance_from_center": lambda a: getattr(a, "distance_from_center", None) if isinstance(a, Land) else None,
        }

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
        self.rent_production = sum(
            agent.model.firm.wage_premium for agent in self.schedule.agents_by_breed[Person].values() 
            if agent.unique_id in agent.workforce.workers
        )

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

    def create_newcomer(self, pos=None):
        """Create newcomer at the center with no residence or property."""
        self.unique_id  += 1

        if pos is None:
            pos = self.center

        savings = self.get_newcomer_init_savings()
        
        person           = Person(self.unique_id, self, pos,
                                  savings = savings,
                                  residence_owned = None)
        self.grid.place_agent(person, pos)
        self.schedule.add(person)
        self.workforce.add(person, self.workforce.newcomers)
        return person

    def get_newcomer_init_savings(self):
        # savings = self.random.uniform(0, 2*self.bank.get_rural_home_value())
        savings = 0.
        return savings

        # DISTRIBUTION FUNCTIONS FOR SAVINGS
        
        # MODEL 1 UNIFORM
        # Max savings/wealth outside the city is twice the value of a rural home
        # Range of savings =  min: 0 max: 2* (a*subsistence_wage/r)

        # Newcomer max could be 1*rural_home_value + age_based_savings
        # Initial resident max  age_based_savings
        # Then we need to set the savings rate to a share of the subsistence wage and 
        # wage_based_savings = savings_rate * subsistence_wage * working_period
        # value of a house with no mortgage = warranted_price: rural_home_value + locational_value

        # MODEL 2 LOGNORMAL
        # from scipy.stats import lognorm
        # # Standard deviation is 1/3 of the value of a rural home
        # stddev = a*subsistence_wage/(3*r) # Standard deviation
        # # Mean is 0.8 the value of a rural home
        # mean = 0.8 * a*subsistence_wage/r # Mean
        # # Draft lognormal function
        # dist=lognorm(a*subsistence_wage/(3*r),loc=0.8 * a*subsistence_wage/r)
        # which should give us a lognorm distribution object with the mean and standard deviation we specify. 
        # We can then inspect the pdf or cdf like this:
        # import numpy as np
        # import pylab as pl
        # x=np.linspace(0,6,200)
        # pl.plot(x,dist.pdf(x))
        # pl.plot(x,dist.cdf(x))

        # MODEL 3 EXPONENTIAL
        # use self.random - np.random.exponential(0.8 * a*subsistence_wage/(r * ln(0.5)), 1)
        # mean for the exponential: 0.8/ln(.5) * rural_home_value
        # draw 1 from this distribution

        # MODEL 4 MIXTURE  
        # Introduce people with debt and debt size E.g. students
        # combine Model  1, 2 or 3 with finite number of students with debt 
        # range of savings = {-100 to 0}

    def get_distance_to_center(self, pos):
        return distance.euclidean(pos, self.center)

class Workforce:
    """Manages a dictionary of working agents."""

    def __init__(self):
        self.workers:               Dict[int, Person] = {}
        self.retiring_urban_owner:  Dict[int, Person] = {}
        self.newcomers:             Dict[int, Person] = {}

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
        self.remove(agent, self.retiring_urban_owner)
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
            self.model.random.shuffle(agent_keys)

        for agent_key in agent_keys:
            agent = agents_dict[agent_key]
            getattr(agent, method)()