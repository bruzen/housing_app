import logging
import os
import yaml
# import functools
import datetime
# import subprocess
# import math
import pandas as pd
from scipy.spatial import distance

from sklearn import linear_model
# from scikit-learn import linear_model
import statsmodels.api as sm
from pysal.model import spreg

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

# def capture_rents(model):
#     """Current rents for each location in the grid."""
#     rent_grid = []
#     for row in model.grid.grid:   
#         new_row = []
#         for cell in row:
#             # TODO Should initial cell_rent be empty/null to signal unassigned?
#             cell_rent = -1.0
#             for item in cell:
#                 try:
#                     # TODO Should we get rent from owner rather than resident?
#                     # TODO Add a warning/error if mult agents have same resid.
#                     if (item.residence):
#                         cell_rent = item.rent
#                 except:
#                     pass
#             new_row.append(cell_rent)
#         rent_grid.append(new_row)
#     return rent_grid

class City(Model):
    # TODO FIX check with agents to confirm
    @property
    def city_extent(self):
        # Compute urban boundary where it is not worthwhile to work
        return ((self.firm.wage - self.subsistence_wage) / 
                 self.transport_cost_per_dist)

    @property
    def r_target(self):
        return self.r_prime + self.r_margin

    def __init__(self, parameters=None):
        super().__init__()

        # Default parameter values
        default_parameters = {
            'run_notes': 'Debugging model.',
            'width': 50,
            'height': 1,
            'init_city_extent': 10.,  # f CUT OR CHANGE?
            'seed_population': 10,
            'density': 100,
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
            #  'wage_adjust_coeff_new_workers': 0.5,
            #  'wage_adjust_coeff_exist_workers': 0.5,
            #  'prefactor': 250,  # CUT, this is A_city? maybe 251, larger than .2
            #  'agglomeration_ratio': 0.12,  # was agglomeration_ratio 1.2,  # CUT? was scaling_factor
            #  'A_F': 53,  # 53.34721 # scale factor for the firm
            #  'A_city': 50,  # prefactor for city
            'mortgage_period': 5.0,  # T, in years
            'working_periods': 40,  # in years
            'savings_rate': 0.3,
            'r_prime': 0.05,  # 0.03
            'r_margin': 0.01,
            'property_tax_rate': 0.04,  # tau, annual rate, was c
            'housing_services_share': 0.3,  # a
            'maintenance_share': 0.2,  # b
            'max_mortgage_share': 0.9,
            'ability_to_carry_mortgage': 0.28,
            'wealth_sensitivity': 0.1,
        }

        # Merge default parameters with provided parameters
        if parameters is not None:
            self.params = {**default_parameters, **parameters}
        else:
            self.params = default_parameters

        # TODO put model name etc. in params
        # TODO maybe remove .self and access params

        # Model
        self.model_name        = 'Housing Market'
        self.model_version     = '0.0.1'
        self.model_description = 'Agent-based housing market model with rent and urban aglomeration.'
        self.run_id    = self.get_run_id(self.model_name, self.model_version)
        self.subfolder = 'output_data' 
        self.time_step = 1.
        self.center = (0, 0) # (width//2, height//2) # TODO make center
        self.grid = MultiGrid(self.params['width'], self.params['height'], torus=False)
        self.schedule = RandomActivationByBreed(self)
        self.seed_population = self.params['seed_population']
        self.density = self.params['density'] # Coarse grain population
        self.transport_cost_per_dist = self.params['init_wage_premium_ratio'] * self.params['subsistence_wage'] / self.params['init_city_extent'] # c
        # self.baseline_population = density*width*height + self.seed_population 

        # People
        self.working_periods = self.params['working_periods']
        self.savings_per_step = self.params['subsistence_wage'] * self.params['savings_rate']
        self.newcomers = []
        self.retiring_agents = []

        # Production model
        self.subsistence_wage = self.params['subsistence_wage'] # psi
        self.workforce_rural_firm = self.params['workforce_rural_firm']
        self.gamma = self.params['gamma']
        self.beta_city = self.params['beta_city']
        self.workers_share = self.params['beta_F'] # lambda

        # Housing market model
        self.mortgage_period = self.params['mortgage_period']
        self.housing_services_share = self.params['housing_services_share'] # a
        self.maintenance_share = self.params['maintenance_share'] # b
        self.r_prime = self.params['r_prime']
        self.r_margin = self.params['r_margin']
        self.discount_factor = self.get_discount_factor()  # sum_delta # TODO - depends on wealth?
        self.max_mortgage_share = self.params['max_mortgage_share']
        self.ability_to_carry_mortgage = self.params['ability_to_carry_mortgage']
        self.wealth_sensitivity = self.params['wealth_sensitivity']
        self.p_dot       = 0. # Price adjustment rate. TODO fix here? rename?
        self.price_model = 0. # TODO need to fix type?

        # Add bank, firm, investor, and realtor
        self.unique_id       = 1
        
        self.bank            = Bank(self.unique_id, self, self.center, self.r_prime)
        self.grid.place_agent(self.bank, self.center)
        self.schedule.add(self.bank)
        
        init_wage_premium = self.params['init_wage_premium_ratio'] * self.params['subsistence_wage'] # omega
        firm_cost_of_capital = self.r_prime
        self.firm            = Firm(self.unique_id, self, self.center, 
                                    init_wage_premium,
                                    self.params['alpha_F'], self.params['beta_F'], self.params['Z'],
                                    self.params['price_of_output'], firm_cost_of_capital,
                                    self.params['wage_adjustment_parameter'],
                                    self.params['firm_adjustment_parameter'])
        self.grid.place_agent(self.firm, self.center)
        self.schedule.add(self.firm)

        self.investor        = Investor(self.unique_id, self, self.center)
        self.grid.place_agent(self.investor, self.center)
        self.schedule.add(self.investor)

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

            init_working_period = self.random.randint(0, 
                                        self.working_periods - 1)
            savings = init_working_period * self.savings_per_step 
            # TODO check boundaries - at working period 0, no savings
            person  = Person(self.unique_id, self, pos,
                                init_working_period = init_working_period,
                                savings             = savings,
                                residence_owned     = land)
            self.grid.place_agent(person, pos)
            self.schedule.add(person)

            self.unique_id  += 1

        # Define what data the model will collect in each time step
        # TODO record interest rate, number of sales, etc.
        model_reporters      = {
#             "rents":          capture_rents,
            "companies":      lambda m: m.schedule.get_breed_count(Firm),
            "people":         lambda m: m.schedule.get_breed_count(Person),
            "wage":           lambda m: m.firm.wage,
            "city_extent":    lambda m: m.city_extent,
            "population":     lambda m: m.firm.N,
            "workers":        lambda m: len(
                [a for a in self.schedule.agents_by_breed[Person].values()
                         if a.is_working == 1]
            )
        }
        agent_reporters      = {
            "agent_class":    lambda a: type(a),
            "id":             lambda a: a.unique_id,
            "x":              lambda a: a.pos[0],
            "y":              lambda a: a.pos[1],
            "is_working":     lambda a: getattr(a, "is_working", None),
            "working_period": lambda a: getattr(a, "working_period", None),
            "property_tax_rate": lambda a: getattr(a, "property_tax_rate", None),
        }

        #     # "wage_premium":   lambda a: getattr(a, "wage_premium", None),
        #     # "rent":         lambda a: getattr(a, "rent", None),
        #     "transport_cost": lambda a: getattr(a, "transport_cost", None),
        #     # "no_workers":     lambda a: getattr(a, "firm_no_workers", None),
        #     "warranted_rent":  lambda a: getattr(a, "warranted_rent", None),
        #     "warranted_price": lambda a: getattr(a, "warranted_price", None),
        #     "net_rent":        lambda a: getattr(a, "net_rent", None),            
        # }
        self.datacollector  = DataCollector(model_reporters = model_reporters,
                                            agent_reporters = agent_reporters)

        # print(vars(self.datacollector))
        # print(dir(self.datacollector))

        # # Check if "Firm" is in the agent types recorded by the data collector
        # if "Firm" in self.datacollector.agent_vars.keys():
        #     print("Init - Firm agent type is recorded in the data collector.")
        # else:
        #     print("InitFirm agent type is not recorded in the data collector.")


        self.step_price_data = [] # for forecasting
        self.price_data = pd.DataFrame(
             columns=['id', 'warranted_price', 'time_step', 'transport_cost', 'wage'])   

        self.record_model_data()

    def record_model_data(self):
        # Create the 'output_data' subfolder if it doesn't exist
        if not os.path.exists(self.subfolder):
            os.makedirs(self.subfolder)

        agent_filename         = self.run_id + '_agent' + '.csv'
        model_filename         = self.run_id + '_model' + '.csv'
        self.agent_file_path   = os.path.join(self.subfolder, agent_filename)
        self.model_file_path   = os.path.join(self.subfolder, model_filename)
        self.metadata_file_path = os.path.join(self.subfolder, 'metadata.yaml')

        metadata = {
            'model_description': self.model_description,
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

    def get_run_id(self, model_name, model_version):
        # Format the current date and time
        current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        # Adapt the model name to lowercase and replace spaces with underscores
        formatted_model_name = model_name.lower().replace(" ", "_")

        # Create the run_id
        return f"{formatted_model_name}_{current_date}_v{model_version.replace('.', '_')}"

    def step(self):
        """ The model step function runs in each time step when the model
        is executed. It calls the agent functions, then records results
        for analysis.
        """

        self.time_step += 1

        logger.debug(f'Step {self.schedule.steps}.')
        self.step_price_data.clear()

        # Land records locational rents and calculates price forecast
        self.schedule.step_breed(Land)
        new_df = pd.DataFrame(self.step_price_data)
        self.price_data = pd.concat([self.price_data, new_df], 
                                          ignore_index=True)

        self.price_model = self.get_price_model()
        self.p_dot       = self.get_p_dot()

        # TODO: if firms hire, should they hire after agents apply, 
        # do they need a seperate function to update wages first?
        # Firms update wages
        
        self.schedule.step_breed(Firm)

        # TODO To speed up, we could calculate rent_growth/history once for 
        # each grid location not each property
        # TODO Or could consider making a land object that has rent, distinct
        # from owned properties
        # for cell in self.grid.coord_iter():
        #     pos              = (cell[1], cell[2])
        #     distance = self.get_distance_to_center(pos)
    
        # People work, retire, and list homes to sell
        self.schedule.step_breed(Person)

        for i in self.retiring_agents:
            # Add agents to replace retiring workers
            person = self.create_newcomer()
            person.bid()

        # Investors bid on properties
        self.schedule.step_breed(Investor)

        # Realtors sell homes
        self.schedule.step_breed(Realtor, step_name='sell_homes')

        # Realtors rent properties
        self.schedule.step_breed(Realtor, step_name='rent_homes')

        # Advance model time
        self.schedule.step_time()

        self.record_step_data()

    def record_step_data(self):

        # Retrieve data
        self.datacollector.collect(self)
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


    def create_newcomer(self):
        """Create newcomer at the center with no residence or property."""
        self.unique_id  += 1
        person           = Person(self.unique_id, self, self.center, 
                                  residence_owned = None)
        self.grid.place_agent(person, self.center)
        self.schedule.add(person)
        self.newcomers.append(person)
        return person

    def get_distance_to_center(self, pos):
        return distance.euclidean(pos, self.center)

# OLD alternative to get_price_model
# def get_rent_growth(self, property):
#     y = np.array(property.rent_history)
#         # Consider the last 5 entries from rent history
#         if len(y) > 5:
#             y = y[-5:]
#         x = np.arange(0,len(y))
#         slope, intercept = np.polyfit(x,y,1)
#         growth_rate = slope 

    def get_price_model(self):
        # TODO 2 models? use realized price, not just warranted
        # Independent variables
        # x = self.price_data[['time_step','transport_cost','wage']]
        x = self.price_data[['time_step']]
        # Dependent variable
        y = self.price_data['warranted_price']

        # with sklearn
        regr = linear_model.LinearRegression()
        regr.fit(x, y)

        # print('Intercept: \n', regr.intercept_)
        # print('Coefficients: \n', regr.coef_)
        # TODO: control for neighbourhood, distance from center, etc.
    
        # # with statsmodels
        # x = sm.add_constant(x) # adding a constant
        
        # model = sm.OLS(y, x).fit()
        # predictions = model.predict(x)   
        return(regr)

    def get_p_dot(self):
        """Rate of growth for property price"""
        # Make p_dot zero for the first 10 steps or so.
        if self.time_step < 10:
            p_dot = 0
        else: 
            # self.p_dot = regr.coef_[0] # slope
            p_dot = self.price_model.coef_[0] # slope
        return p_dot

    # Workers share is lambda- workers share of the surplus. DO NOT USE THIS. JUST COMPUTE A VALUE FOR LAMBRA - MAYBE 0.8
    # def get_workers_share(self):
    #     """Share of wage premium that goes to workers, omega """
    #     psi     = self.subsistence_wage
    #     lambda  = self.workers_share # now beta_F
    #     agglom  = self.agglomeration_ratio
    #     return lambda * agglom * psi

    # TODO could vary with wealth
    def get_discount_factor(self):
        """
        The discount factor gives the present value of one dollar received at particular point in the future, given the date of receipt and the discount rate.
        Delta is the subjective individual discount rate for agent
        after one year. This will be close to the prime interest rate, r_i.
        """    
        delta = self.r_prime
        delta_period_1 = 1 / (1 + delta) 
        delta_mortgage_period = delta_period_1**self.mortgage_period
        sum_delta = (1 - delta_mortgage_period)/delta
        return sum_delta
        # sum_delta = delta_mortgage_period * (1 - delta_mortgage_period) # Old