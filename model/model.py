import logging
import os

from typing import Dict, List
from scipy.spatial import distance

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import model.parameters as params
import utils.file_utils as file_utils
from model.agents import Land, Person, Firm, Investor, Bank, Realtor
from model.schedule import RandomActivationByBreed

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

        default_parameters = params.default_parameters

        # Merge default parameters with provided parameters
        if parameters is not None:
            self.params = {**default_parameters, **parameters}
        else:
            self.params = default_parameters

        # Model
        self.model_name        = 'Main' # 'Housing Market'
        self.model_version     = '0.1.0'
        self.model_description = 'Agent-based housing market model with rent and urban agglomeration.'
        self.num_steps = num_steps

        # Interventions
        if 'intervention' in self.params and self.params['intervention'] is True:
            self.intervention = True
            if 'perturb_at_time' in self.params:
                try:
                    perturb_at_time_data = self.params['perturb_at_time']

                    # if not isinstance(perturb_at_time_data, dict):
                    #     raise TypeError("'perturb_at_time' should be a dictionary.")

                    self.perturb_var  = perturb_at_time_data.get('var', None)
                    self.perturb_val  = perturb_at_time_data.get('val', None)
                    self.perturb_time = perturb_at_time_data.get('time', None)

                    if any(var is None for var in (self.perturb_var, self.perturb_val, self.perturb_time)):
                        raise ValueError("Invalid or missing data in 'perturb_at_time'.")
                except Exception as e:
                    print(f"An error occurred with processing perturb_at_time: {e}")
        else:
            self.intervention = False # TODO Can use to control the logic for any intervention

        self.setup_run_data_collection()

        # Record metadata
        self.metadata  = file_utils.record_metadata(self, filepath = self.metadata_filepath)

        logging.basicConfig(filename=self.log_filepath,
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s:%(message)s')
        self.logger = logging.getLogger(__name__)

        # Set logger level for the matplotlib logger, to reduce outside log messages
        logging.getLogger('matplotlib').setLevel(logging.ERROR)

        # Initialize interventions if interventions_on is True, and interventsions is a non empty dict
        if 'interventions' in self.params and self.params.get('interventions_on', False):
            self.interventions = self.params['interventions']
            if not self.interventions:  # Check if interventions is an empty dictionary
                self.interventions = None
                # self.logger.warning("Empty interventions provided.")
        else:
            self.interventions = None
            # self.logger.warning("No interventions provided.")

        # Initialize counters
        self.urban_investor_owners_count = 0
        self.urban_resident_owners_count = 0
        self.urban_other_owners_count    = 0

        # # Set the random seed for reproducibility
        # self.random_seed = 42
        # self.random.seed(self.random_seed)
        # current_time_seed = int(time.time())
        # random.seed(current_time_seed)

        # Setup grid
        self.height = self.params['height']
        self.width  = self.params['width']
        # If self.center_city is True, it places the city in the center; otherwise, it places it in the bottom corner.
        self.center_city   = self.params['center_city'] # put city in the bottom corner TODO check flag's logic
        if self.center_city:
            self.center    = (width//2, height//2)
        else:
            self.center    = (0, 0)
        self.grid = MultiGrid(self.params['width'], self.params['height'], torus=False)
        self.schedule = RandomActivationByBreed(self)
        self.transport_cost_per_dist = self.params['c'] # self.params['init_wage_premium_ratio'] * self.params['subsistence_wage'] / self.params['init_city_extent'] # c

        # People
        # If demographics_on, there is a housing market when agents retire # self.demographics_on = self.params['demographics_on']
        if self.params['demographics_on']:
            self.working_periods  = self.params['working_periods']
            self.logger.debug(f'Demographics on, working periods {self.working_periods}, time steps {self.num_steps}') #, params working periods {self.params['working_periods']}')
        else:
            self.working_periods = 10 * self.num_steps
            self.logger.debug(f'Demographics off, working periods {self.working_periods}')
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

        # Add class to track retired agents that still own property
        self.unique_id       = 1
        self.retired_agents  = Retired_Agents(self, self.unique_id)

        # Add bank, firm, investor, and realtor
        self.unique_id      += 1
        self.bank            = Bank(self.unique_id, self, self.center)
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
                                    adjN=self.params['adjN'],
                                    adjk=self.params['adjk'],
                                    adjn=self.params['adjn'],
                                    adjF=self.params['adjF'],
                                    adjw=self.params['adjw'],
                                    adjs=self.params['adjs'],
                                    adjd=self.params['adjd'],
                                    adjp=self.params['adjp'],
                                    dist=self.params['dist'],
                                    init_F=self.params['init_F'],
                                    init_k=self.params['init_k'],
                                    init_n=self.params['init_n'],
                                    )
        self.grid.place_agent(self.firm, self.center)
        self.schedule.add(self.firm)

        self.unique_id      += 1
        self.investor        = Investor(self.unique_id, self, self.center, self.params['r_investor'], self.params['cg_tax_invest'], self.params['investor_expectations'], self.params['investor_turnover'])
        self.grid.place_agent(self.investor, self.center)
        self.schedule.add(self.investor)

        self.unique_id      += 1
        self.realtor         = Realtor(self.unique_id, self, self.center)
        self.grid.place_agent(self.realtor, self.center)
        self.schedule.add(self.realtor)

        # Add land and people to each cell
        self.unique_id      += 1
        for cell in self.grid.coord_iter():
            pos              = (cell[1][0], cell[1][1])

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
                                capital_gains_tax   = self.params['cg_tax_per'],
                                residence_owned     = land)
            self.grid.place_agent(person, pos)
            self.schedule.add(person)

            self.unique_id  += 1

        self.setup_mesa_data_collection()
        # self.record_step_data()

        # Run the firm for several steps to stabilize
        for i in range(2):
            # People check if it's worthwhile to work
            self.schedule.step_breed(Person, step_name='work_if_worthwhile_to_work')

            # Firms update wages
            self.schedule.step_breed(Firm)

    def step(self):
        """ The model step function runs in each time step when the model
        is executed. It calls the agent functions, then records results
        for analysis.
        """

        # Apply interventions if there are any
        if self.interventions:
            self.apply_interventions(self.schedule.time)

        # Reset counters
        self.urban_investor_owners_count = 0
        self.urban_resident_owners_count = 0
        self.urban_other_owners_count    = 0

        # self.logger.info(f'\n \n \n Step {self.schedule.steps}. \n')
        self.logger.info(f'\n \n \n Step {self.schedule.time}. \n')

        # Firms update wages based on how many people choose to work in the city
        self.firm.worker_supply = self.firm.get_worker_supply()
        self.firm.agglom_pop    = self.firm.get_agglomeration_population(self.firm.worker_supply)
        self.schedule.step_breed(Firm)

        # Land records locational rents and calculates price forecast
        self.schedule.step_breed(Land)
    
        # People work, retire, and list properties to sell
        self.schedule.step_breed(Person)

        # Investors list properties to sell
        self.schedule.step_breed(Investor, step_name='list_properties')

        # Add agents to replace retiring_urban_owner workers
        # Draw 50 values from the distribution of initial savings
        savings_values = [self.get_newcomer_init_savings() for _ in range(50)]

        # Sort the list in descending order
        savings_values.sort(reverse=True)
        # self.logger.debug(f'Savings value distribution {savings_values}')

        for i in self.workforce.retiring_urban_owner:
            if savings_values:
                top_savings = savings_values.pop()
                # top_savings = 0.
                person = self.create_newcomer(top_savings)
                person.bid_on_properties()
            else:
                # Handle the case where there are no more savings values
                # You might want to break out of the loop or take other appropriate actions
                self.logger.error(f'Did not create newcomer since not enough savings values.')

        # Investors bid on properties
        self.schedule.step_breed(Investor, step_name='bid_on_properties')

        # Realtors sell homes
        self.schedule.step_breed(Realtor, step_name='sell_homes')

        # Realtors rent properties
        self.schedule.step_breed(Realtor, step_name='rent_homes')

        # Advance model time
        self.schedule.step_time()

        # Do checks TEMP
        for cell in self.grid.coord_iter():
            pos                 = (cell[1][0], cell[1][1])
            cell_contents       = self.grid.get_cell_list_contents(pos)
            persons_at_position = [agent for agent in cell_contents if isinstance(agent, Person)]

            if len(persons_at_position) > 1:
                num_not_in_newcomers = [agent for agent in persons_at_position if agent not in self.workforce.newcomers]
                if len(num_not_in_newcomers) > 1:
                    # Extra non-newcomers
                    unique_ids = [str(agent.unique_id) for agent in num_not_in_newcomers]
                    comma_separated_unique_ids = ', '.join(unique_ids)
                    self.logger.warning(f'More than one Person agent in self.workforce.newcomers at location {pos}, agents are {comma_separated_unique_ids}')
                elif len(num_not_in_newcomers == 0):
                    # Only newcomers
                    self.logger.warning(f'Only newcomers at location {pos}, agents are {comma_separated_unique_ids}')
                # else:
                #     # Newcomers
                #     unique_ids = [str(agent.unique_id) for agent in persons_at_position]
                #     comma_separated_unique_ids = ', '.join(unique_ids)
                #     self.logger.debug(f'More than one Person agent at location {pos}, agents are {comma_separated_unique_ids}')
                # TODO Could check that non-newcomer is resident etc.

            elif len(persons_at_position) == 0:
                # There are no people
                self.logger.warning(f'No Person agents at location {pos}')

        self.record_step_data()

    def run_model(self):
        for t in range(self.num_steps):
            self.step()

        self.record_run_data_to_file()

    # TODO consider moving to file_utils
    def setup_run_data_collection(self):
        # Set timestamp and run_id
        if 'timestamp' in self.params and self.params['timestamp'] is not None:
            self.timestamp     = self.params['timestamp']
        else:
            self.timestamp     = file_utils.generate_timestamp()
        self.run_id            = file_utils.get_run_id(self.timestamp) #, self.model_name, self.model_version)

        # Set log and metadata filepaths
        self.log_filepath      = file_utils.get_log_filepath(file_name = f'log-{self.timestamp}.log')
        self.metadata_filepath = file_utils.get_metadata_filepath(file_name = f'metadata-{self.run_id}.json')

        # # Set figures folder
        # self.figures_folder  = file_utils.get_figures_subfolder()

        # Set data filepaths
        if 'subfolder' in self.params and self.params['subfolder'] is not None:
            self.data_folder   = self.params['subfolder']
        else:
            self.data_folder   = file_utils.get_data_subfolder()
        self.agent_filepath    = os.path.join(self.data_folder, f"data-{self.run_id}-agent.csv")
        self.model_filepath    = os.path.join(self.data_folder, f"data-{self.run_id}-model.csv")

    def setup_mesa_data_collection(self):
        self.store_agent_data = self.params['store_agent_data']
        self.no_decimals      = self.params['no_decimals']
        model_reporters       = {
            "model_name":                lambda m: m.model_name,
            "run_id":                    lambda m: m.run_id,
            "time_step":                 lambda m: m.schedule.time,
            "MPL":                       lambda m: round(m.firm.MPL, self.no_decimals),
            "city_extent_calc":          lambda m: round(m.city_extent_calc, self.no_decimals),
            "n":                         lambda m: round(m.firm.n, self.no_decimals),
            "y":                         lambda m: round(m.firm.y, self.no_decimals),
            "F_target":                  lambda m: round(m.firm.F_target, self.no_decimals),
            "F":                         lambda m: round(m.firm.F, self.no_decimals),
            "k":                         lambda m: round(m.firm.k, self.no_decimals),
            "N":                         lambda m: round(m.firm.N, self.no_decimals),
            "wage":                      lambda m: round(m.firm.wage, self.no_decimals),
            # "N/F":                     lambda m: round(m.firm.N/m.firm.F, self.no_decimals),
            "wage":                      lambda m: round(m.firm.wage, self.no_decimals),
            "subsistence_wage":          lambda m: round(m.firm.subsistence_wage, self.no_decimals),
            "wage_target":               lambda m: round(m.firm.wage_target, self.no_decimals),
            "worker_supply":             lambda m: round(m.firm.worker_supply, self.no_decimals),
            "worker_demand":             lambda m: round(m.firm.worker_demand, self.no_decimals),
            "agglomeration_population":  lambda m: round(m.firm.agglom_pop, self.no_decimals),
        }
        # # Define what data the model will collect in each time step
        # model_reporters = {
        #     "model_name":                lambda m: m.model_name,
        #     "run_id":                    lambda m: m.run_id,
        #     "workers":                   lambda m: m.firm.N,
        #     "MPL":                       lambda m: m.firm.MPL,
        #     "companies":                 lambda m: m.schedule.get_breed_count(Firm),
        #     "city_extent_calc":          lambda m: m.city_extent_calc,
        #     "people":                    lambda m: m.schedule.get_breed_count(Person),
        #     "market_rent":               lambda m: m.market_rent,
        #     "net_rent":                  lambda m: m.net_rent,
        #     "potential_dissipated_rent": lambda m: m.potential_dissipated_rent,
        #     "dissipated_rent":           lambda m: m.dissipated_rent,
        #     "available_rent":            lambda m: m.available_rent,
        #     "rent_captured_by_finance":  lambda m: m.rent_captured_by_finance,
        #     "share_captured_by_finance": lambda m: m.share_captured_by_finance,
        #     "urban_surplus":             lambda m: m.urban_surplus,
        #     "removed_agents":            lambda m: m.removed_agents,
        #     "n":                         lambda m: m.firm.n,
        #     "y":                         lambda m: m.firm.y,
        #     "F_target":                  lambda m: m.firm.F_target,
        #     "F":                         lambda m: m.firm.F,
        #     "k":                         lambda m: m.firm.k,
        #     "N":                         lambda m: m.firm.N,
        #     # "agglomeration_population":  lambda m: m.firm.agglomeration_population, # TODO delete
        #     "Y":                         lambda m: m.firm.Y,
        #     "wage_premium":              lambda m: m.firm.wage_premium,
        #     "subsistence_wage":          lambda m: m.firm.subsistence_wage,
        #     "wage":                      lambda m: m.firm.wage,
        #     "wage_target":               lambda m: m.firm.wage_target,
        #     # "worker_agents":           lambda m: m.workforce.get_agent_count(m.workforce.workers),
        #     "worker_agents":             lambda m: len(m.workforce.workers),
        #     "newcomer_agents":           lambda m: len(m.workforce.newcomers),
        #     "retiring_urban_owner":      lambda m: len(m.workforce.retiring_urban_owner),
        #     "urban_resident_owners":     lambda m: m.urban_resident_owners_count,
        #     "urban_investor_owners":     lambda m: m.urban_investor_owners_count,
        #     "urban_other_owners":        lambda m: m.urban_other_owners_count,
        #     "investor_ownership_share":  lambda m: m.urban_investor_owners_count / (m.urban_resident_owners_count + m.urban_investor_owners_count) if (m.urban_resident_owners_count + m.urban_investor_owners_count) != 0 else 1,
        #     # "workers":        lambda m: len(
        #     #     [a for a in self.schedule.agents_by_breed[Person].values()
        #     #              if a.is_working == 1]
        #     # )
        # }

        if self.store_agent_data:
            # Variables for data collection
            self.rent_production   = 0.
            self.rent_amenity      = 0.
            self.market_rent       = 0.
            self.net_rent          = 0.
            self.potential_dissipated_rent = 0.
            self.dissipated_rent   = 0.
            self.available_rent    = 0.
            self.rent_captured_by_finance  = 0.
            self.share_captured_by_finance = 0.
            self.urban_surplus     = 0.

            agent_reporters      = {
                "time_step":         lambda a: a.model.schedule.time,
                "agent_class":       lambda a: type(a),
                "agent_type":        lambda a: type(a).__name__,
                "id":                lambda a: a.unique_id,
                "x":                 lambda a: a.pos[0],
                "y":                 lambda a: a.pos[1],
                "is_working":        lambda a: None if not isinstance(a, Person) else 1 if a.unique_id in a.model.workforce.workers else 0,  # TODO does this need to be in model? e.g. a.model.workforce
                "is_working_check":  lambda a: None if not isinstance(a, Person) else a.is_working_check,
                "working_period":    lambda a: getattr(a, "working_period", None)  if isinstance(a, Person)       else None,
                "p_dot":             lambda a: getattr(a, "p_dot", None)           if isinstance(a, Land)         else None,
                "net_rent":          lambda a: getattr(a, "net_rent", None)        if isinstance(a, Land)         else None,
                "warranted_rent":    lambda a: getattr(a, "warranted_rent", None)  if isinstance(a, Land)         else None,
                "warranted_price":   lambda a: getattr(a, "warranted_price", None) if isinstance(a, Land)         else None,
                "realized_price":    lambda a: getattr(a, "realized_price", None)  if isinstance(a, Land)         else None,
                "realized_all_steps_price": lambda a: getattr(a, "realized_all_steps_price", None)  if isinstance(a, Land) else None,
                "sold_this_step":    lambda a: getattr(a, "sold_this_step", None)  if isinstance(a, Land)         else None,
                "ownership_type":    lambda a: getattr(a, "ownership_type", None)  if isinstance(a, Land)         else None,
                "distance_from_center": lambda a: getattr(a, "distance_from_center", None) if isinstance(a, Land) else None,
            }
            self.datacollector  = DataCollector(model_reporters = model_reporters,
                                                agent_reporters = agent_reporters)
        else:
            self.datacollector  = DataCollector(model_reporters = model_reporters)

    def record_step_data(self):
        # Calculations for data collection
        self.rent_production = sum(
            agent.model.firm.wage_premium for agent in self.schedule.agents_by_breed[Person].values() 
            if agent.unique_id in self.workforce.workers
        )

        self.rent_amenity    = sum(
            agent.amenity for agent in self.schedule.agents_by_breed[Person].values() 
            if agent.unique_id in self.workforce.workers
        )

        self.market_rent = sum(agent.market_rent    for agent in self.schedule.agents_by_breed[Land].values()
                               if agent.resident and agent.resident.unique_id in self.workforce.workers)
        self.net_rent    = sum(agent.net_rent       for agent in self.schedule.agents_by_breed[Land].values()
                               if agent.resident and agent.resident.unique_id in self.workforce.workers)
        self.potential_dissipated_rent = sum(agent.transport_cost for agent in self.schedule.agents_by_breed[Land].values())
        self.dissipated_rent = sum(
            agent.transport_cost for agent in self.schedule.agents_by_breed[Land].values() 
            if agent.resident and agent.resident.unique_id in self.workforce.workers
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
                agent_out.to_csv(self.agent_filepath, index=False)
            except Exception as e:
                self.logger.error("Error saving agent data: %s", str(e))

        # Save model data
        if model_out is not None:
            try:
                model_out.to_csv(self.model_filepath, index=False)
            except Exception as e:
                self.logger.error("Error saving model data: %s", str(e))


    def create_newcomer(self, savings=0, pos=None):
        """Create newcomer at the center with no residence or property."""
        self.unique_id  += 1

        if pos is None:
            pos = self.center
        
        person  = Person(self.unique_id, self, pos,
                  savings           = savings,
                  capital_gains_tax = self.params['cg_tax_per'],
                  residence_owned   = None)
        self.grid.place_agent(person, pos)
        self.schedule.add(person)
        self.workforce.add(person, self.workforce.newcomers)
        self.logger.debug(f'Newcomer savings {person.unique_id}, {savings}')
        return person

    def get_newcomer_init_savings(self):
        savings = self.random.uniform(0, 2*self.bank.get_rural_home_value())
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

    def apply_interventions(self, current_time_step):
        # Check if any interventions match the current time step
        for intervention_name, intervention_details in self.interventions.items():
            if current_time_step == intervention_details['time']:

                # Split the attribute path into its components
                attr_components = intervention_details['var'].split('.')
                target_obj = self
                for attr_name in attr_components[:-1]:
                    target_obj = getattr(target_obj, attr_name)

                # Set the value of the final attribute, print before and after
                print(f"{intervention_name} at time {current_time_step}:")
                print(f"   Before change, value is {getattr(target_obj, attr_components[-1])}")
                setattr(target_obj, attr_components[-1], intervention_details['val'])
                print(f"   After change, value is  {getattr(target_obj, attr_components[-1])}, at time {current_time_step} \n")

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
    #         self.logger.warning(f"Person with unique id {agent.unique_id!r} is already in the manager.")
    #         return  # If agent is already present, simply return without adding it again.

    #     agent_dict[agent.unique_id] = agent

    # def remove(self, agent: Person, agents_dict: Dict[int, Person]) -> None:
    #     if agent.unique_id not in agents_dict:
    #         self.logger.warning(f"Person with unique id {agent.unique_id!r} not found in the manager")
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

class Retired_Agents:
    def __init__(self, model, unique_id):
        self.model              = model
        self.unique_id          = unique_id
        self.property_ownership = {}

    def add_property(self, owner_id, property):
        if owner_id not in self.property_ownership:
            # If the agent is not already in the dictionary, add them with the new property
            self.property_ownership[owner_id] = {
                "properties": [property]
            }
        else:
            # If the agent is already in the dictionary, append the new property to the existing list
            self.property_ownership[owner_id]["properties"].append(property)
            self.model.logger(f'Agent_id {owner_id} already exists in Retired_Agents. Added property to list of properties owned.')