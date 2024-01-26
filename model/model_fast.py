import logging
import os
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

import model.parameters as params
import utils.file_utils as file_utils
from model.agents import Land, Person, Firm, Investor, Bank, Realtor, Bid_Storage
from model.schedule import RandomActivationByBreed

class Fast(Model):
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
        self.model_name        = 'Housing Market'
        self.model_version     = '0.0.1'
        self.model_description = 'Agent-based housing market model with rent and urban agglomeration.'
        self.num_steps = num_steps
        # self.time_step = 0

        self.setup_run_data_collection()

        # Record metadata
        self.metadata  = file_utils.record_metadata(filepath = self.metadata_filepath, run_id = self.run_id, num_steps = self.num_steps, params=self.params)

        logging.basicConfig(filename=self.log_filepath,
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s:%(message)s')
        self.logger = logging.getLogger(__name__)

        logging.getLogger('matplotlib').setLevel(logging.ERROR)

        self.height = self.params['height']
        self.width  = self.params['width']

        # Initialize counters
        self.urban_investor_owners_count = 0
        self.urban_resident_owners_count = 0
        self.urban_other_owners_count    = 0

        # # Set the random seed for reproducibility
        # self.random_seed = 42
        # self.random.seed(self.random_seed)

        # current_time_seed = int(time.time())
        # random.seed(current_time_seed)

        # # If self.center_city is True, it places the city in the center; otherwise, it places it in the bottom corner.
        self.center_city   = self.params['center_city'] # put city in the bottom corner TODO check flag's logic
        # if self.center_city:
        #     self.center    = (width//2, height//2)
        # else:
        self.center    = (0, 0)
        self.grid = MultiGrid(self.params['width'], self.params['height'], torus=False)
        self.schedule = RandomActivationByBreed(self)
        self.transport_cost_per_dist = self.params['c'] # self.params['init_wage_premium_ratio'] * self.params['subsistence_wage'] / self.params['init_city_extent'] # c

        # People
        # If demographics_on, there is a housing market when agents retire # self.demographics_on = self.params['demographics_on']
        if self.params['demographics_on']:
            self.working_periods  = self.params['working_periods']
            # self.logger.debug(f'Demographics on, working periods {self.working_periods}, 2x time steps {self.num_steps}') #, params working periods {self.params['working_periods']}')
        else:
            self.working_periods = 10 * self.num_steps
            # self.logger.debug(f'Demographics off, working periods {self.working_periods}')
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
        # self.workforce = Workforce()
        self.removed_agents = 0

        # Add class to track retired agents that still own property
        self.unique_id       = 1
        # self.retired_agents  = Retired_Agents(self, self.unique_id)

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
                                    dist=self.params['dist'],
                                    init_F=self.params['init_F'],
                                    init_k=self.params['init_k'],
                                    init_n=self.params['init_n']
                                    )
        self.grid.place_agent(self.firm, self.center)
        self.schedule.add(self.firm)

        self.unique_id      += 1
        self.investor        = Investor(self.unique_id, self, self.center, self.params['r_investor'], self.params['cg_tax_invest'])
        self.grid.place_agent(self.investor, self.center)
        self.schedule.add(self.investor)

        self.unique_id      += 1
        self.realtor         = Realtor(self.unique_id, self, self.center)
        self.grid.place_agent(self.realtor, self.center)
        self.schedule.add(self.realtor)

        self.unique_id      += 1
        self.person  = Person(self.unique_id, self, self.center,
                            init_working_period = 0,
                            savings             = 0,
                            capital_gains_tax   = self.params['cg_tax_per'],
                            residence_owned     = None)
        self.grid.place_agent(self.person, self.center)
        self.schedule.add(self.person)

        self.unique_id      += 1
        self.property        = Land(self.unique_id, self, self.center, 
                                    self.params['property_tax_rate'])
        self.grid.place_agent(self.property, self.center)
        self.schedule.add(self.property)

        # Select the distances at which to record values
        if self.params['distances']:
            self.distances = distances
        else:
            max_radius = 150
            sparseness = 40
            distances = int(max_radius / sparseness)

        # Determine the initial savings values to evaluate
        if self.params['newcomer_savings']:
            self.newcomer_savings = newcomer_savings
        else:
            min_savings = 0
            max_savings = 2*self.bank.get_rural_home_value()
            no_steps = 3
            step_size = int((max_savings - min_savings) / (no_steps - 1))
            newcomer_savings = [min_savings + i * step_size for i in range(no_steps)]
            # print(f'Newcomer savings: {self.newcomer_savings}')

        # Add Bid_Storage to store bids
        for x in range(distances):
            self.unique_id      += 1
            dist = x * sparseness
            bidder_name = 'Investor'         
            agent       = Bid_Storage(self.unique_id, self, self.center, 
                                        bidder_name          = bidder_name,
                                        distance_from_center = dist)
            self.grid.place_agent(agent, self.center)
            self.schedule.add(agent)      
            for y, savings_val in enumerate(newcomer_savings):
                self.unique_id      += 1
                bidder_name = f'Savings {savings_val}'
                agent   = Bid_Storage(self.unique_id, self, self.center, 
                                      bidder_name          = bidder_name,
                                      distance_from_center = dist,
                                      bidder_savings       = savings_val)
                self.grid.place_agent(agent, self.center)
                self.schedule.add(agent)
                
        # # Add land and people to each cell
        # self.unique_id      += 1
        # for cell in self.grid.coord_iter():
        #     pos              = (cell[1][0], cell[1][1])

        #     land             = Land(self.unique_id, self, pos, 
        #                             self.params['property_tax_rate'])
        #     self.grid.place_agent(land, pos)
        #     self.schedule.add(land)

        #     self.unique_id      += 1
        #     # TODO maybe control flow for this with a flag passed in
        #     if self.params['random_init_age']:
        #         init_working_period  = self.random.randint(0, self.params['working_periods'] - 1) # TODO randomize working period
        #     else:
        #         init_working_period  = 0
        #     savings = init_working_period * self.savings_per_step 
        #     # TODO check boundaries - at working period 0, no savings
        #     person  = Person(self.unique_id, self, pos,
        #                         init_working_period = init_working_period,
        #                         savings             = savings,
        #                         capital_gains_tax   = self.params['cg_tax_per'],
        #                         residence_owned     = land)
        #     self.grid.place_agent(person, pos)
        #     self.schedule.add(person)

        #     self.unique_id  += 1

        # Init for step_fast
        # Create a list of savings levels representing newcomers who will bid

        # self.step_data = {
        #     "dist":            [],
        #     "m":               [],
        #     "R_N":             [],
        #     "p_dot":           [],
        #     "transport_cost":  [],
        #     "investor_bid":    [],
        #     "warranted_rent":  [],
        #     "warranted_price": [],
        #     "maintenance":     [],
        #     "newcomer_bid":    [],
        # }

        self.no_decimals = 1
        self.setup_mesa_data_collection()
        # self.record_step_data()

        # # Run the firm for several steps to stabilize
        # for i in range(2):
        #     # People check if it's worthwhile to work
        #     self.schedule.step_breed(Person, step_name='work_if_worthwhile_to_work')

        #     # Firms update wages
        #     self.schedule.step_breed(Firm)   

    def step(self):
       
        # Firm updates wages based on agglomeration population
        self.schedule.step_breed(Firm)

        # Firm updates agglomeration population based on calculated city extent
        extent = self.city_extent_calc
        self.firm.N = self.firm.get_N_from_city_extent(extent)

        self.schedule.step_breed(Bid_Storage)

        self.datacollector.collect(self)
        self.schedule.step_time()        
 
        # print('Step')
        # while self.time_step < 10:
        #     self.time_step += 1
        #     self.reset_step_data_lists()

        #     print(self.time_step)
        #     # Firm updates wages based on agglomeration population
        #     # self.firm.step()
        #     self.schedule.step_breed(Firm)

        #     # Firm updates agglomeration population based on calculated city extent
        #     extent = self.city_extent_calc
        #     self.firm.N = self.firm.get_N_from_city_extent(extent)

        #     self.schedule.step_breed(Bid_Storage)
            # # Calculate bid_rent values function of distance and person's savings
            # # TODO does this exclude some of the city, effectively rounding down? Do rounding effects matter for the city extent/population calculations?
            # # dist = 0
            # # while dist <= extent:
            # num_steps = 3
            # step_size = extent // (num_steps - 1) if num_steps > 1 else 1  # Calculate the step size

            # # TODO REPLICATE THIS LOGIC
            # for step in range(num_steps):
            #     dist = step * step_size
            #     m       = self.max_mortgage_share
            #     self.property.change_dist(dist)

            #     R_N             = self.property.net_rent
            #     p_dot           = self.property.p_dot
            #     transport_cost  = self.property.transport_cost
            #     investor_bid,  investor_bid_type = self.investor.get_max_bid(m = m,
            #                                     R_N   = R_N,
            #                                     p_dot = p_dot,
            #                                     transport_cost = transport_cost)

            #     warranted_rent  = self.property.get_warranted_rent()
            #     warranted_price = self.property.get_warranted_price()
            #     maintenance     = self.property.get_maintenance()

            #     attributes_to_append = ["dist", "m", "R_N", "p_dot", "transport_cost", "investor_bid", "warranted_rent", "warranted_price", "maintenance"]
            #     for attribute in attributes_to_append:
            #         value = locals()[attribute]  # Get the value of the attribute
            #         # self.step_data[attribute].append((value, dist))
            #         self.step_data[attribute].append(round(value, self.no_decimals))

            #     # print(f'bid {investor_bid}, m {m}, R_N {R_N}, p_Dot {p_dot}, transp {transport_cost}')
            #     # print(f'Property dist {self.property.distance_from_center}, transport_cost {self.property.transport_cost}, i_bid {investor_bid} {investor_bid_type}')

            #     for savings_value in self.newcomer_savings:
            #         # Calculate newcomers bid
            #         M     = self.person.get_max_mortgage(savings_value)
            #         newcomer_bid,  newcomer_bid_type = self.person.get_max_bid(m, M, R_N, p_dot, transport_cost, savings_value)
            #         self.step_data["newcomer_bid"].append((round(newcomer_bid, self.no_decimals), round(dist, self.no_decimals), round(savings_value, self.no_decimals)))
            #     # dist += 1

    def setup_run_data_collection(self):
        # Set timestamp and run_id
        if 'timestamp' in self.params and self.params['timestamp'] is not None:
            self.timestamp     = self.params['timestamp']
        else:
            self.timestamp     = file_utils.generate_timestamp()
        self.run_id            = file_utils.get_run_id(self.timestamp) # self.model_name, self.model_version)

        # Set log and metadata filepaths
        self.log_filepath      = file_utils.get_filepath(folder_name = "output_data", subfolder_name = "logs", file_name = f'fast-log-{self.timestamp}.log')
        self.metadata_filepath = file_utils.get_filepath(folder_name = "output_data", subfolder_name = "metadata", file_name = f'fast-metadata-{self.run_id}.log')

        # # Set figures folder
        # self.figures_folder = file_utils.get_subfolder(folder_name = "output_data", subfolder_name = "figures")

        # Set data filepaths
        if 'subfolder' in self.params and self.params['subfolder'] is not None:
            self.data_folder = self.params['subfolder']
        else:
            self.data_folder = file_utils.get_subfolder(folder_name = "output_data", subfolder_name = "data")
        self.agent_filepath   = os.path.join(self.data_folder, f"{self.run_id}-agent.csv")
        self.model_filepath   = os.path.join(self.data_folder, f"{self.run_id}-model.csv")

    def setup_mesa_data_collection(self):

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

        # Define what data the model will collect in each time step
        model_reporters      = {
            "run_id":                      lambda m: m.run_id,
            # "workers":                   lambda m: m.firm.N,
            "MPL":                         lambda m: m.firm.MPL,
            "time_step":                   lambda m: m.schedule.time,
            # "companies":                 lambda m: m.schedule.get_breed_count(Firm),
            "city_extent_calc":            lambda m: round(m.city_extent_calc, self.no_decimals),
            # # "people":                    lambda m: m.schedule.get_breed_count(Person),
            # # "market_rent":               lambda m: m.market_rent,
            # # "net_rent":                  lambda m: m.net_rent,
            # # "potential_dissipated_rent": lambda m: m.potential_dissipated_rent,
            # # "dissipated_rent":           lambda m: m.dissipated_rent,
            # # "available_rent":            lambda m: m.available_rent,
            # # "rent_captured_by_finance":  lambda m: m.rent_captured_by_finance,
            # # "share_captured_by_finance": lambda m: m.share_captured_by_finance,
            # # "urban_surplus":             lambda m: m.urban_surplus,
            # # "removed_agents":            lambda m: m.removed_agents,
            "n":                         lambda m: round(m.firm.n, self.no_decimals),
            "y":                         lambda m: round(m.firm.y, self.no_decimals),
            "F_target":                  lambda m: round(m.firm.F_target, self.no_decimals),
            "F":                         lambda m: round(m.firm.F, self.no_decimals),
            "k":                         lambda m: round(m.firm.k, self.no_decimals),
            "N":                         lambda m: round(m.firm.N, self.no_decimals),
            "N/F":                       lambda m: round(m.firm.N/m.firm.F, self.no_decimals),
            # # # "agglomeration_population":  lambda m: m.firm.agglomeration_population, # TODO delete
            # # "Y":                         lambda m: m.firm.Y,
            # "wage_premium":              lambda m: round(m.firm.wage_premium, self.no_decimals),
            # "p_dot":                     lambda m: round(m.firm.p_dot, self.no_decimals),
            # # "subsistence_wage":          lambda m: m.firm.subsistence_wage,
            # # "wage":                      lambda m: m.firm.wage,
            # # # "worker_agents":           lambda m: m.workforce.get_agent_count(m.workforce.workers),
            # # "worker_agents":             lambda m: len(m.workforce.workers),
            # # "newcomer_agents":           lambda m: len(m.workforce.newcomers),
            # # "retiring_urban_owner":      lambda m: len(m.workforce.retiring_urban_owner),
            # # "urban_resident_owners":     lambda m: m.urban_resident_owners_count,
            # # "urban_investor_owners":     lambda m: m.urban_investor_owners_count,
            # # "urban_other_owners":        lambda m: m.urban_other_owners_count,
            # # "investor_ownership_share":  lambda m: m.urban_investor_owners_count / (m.urban_resident_owners_count + m.urban_investor_owners_count) if (m.urban_resident_owners_count + m.urban_investor_owners_count) != 0 else 1,
            # # "workers":        lambda m: len(
            # #     [a for a in self.schedule.agents_by_breed[Person].values()
            # #              if a.is_working == 1]
            # # )
            #     # "investor_bid":    lambda m: m.step_data["investor_bid"],
            #     # "warranted_rent":  lambda m: m.step_data["warranted_rent"],
            #     # "warranted_price": lambda m: m.step_data["warranted_price"],
            #     # "dist":            lambda m: m.step_data["dist"],
            #     # # "m":               lambda m: m.step_data["m"],
            #     # "R_N":             lambda m: m.step_data["R_N"],
            #     # # "p_dot":           lambda m: m.step_data["p_dot"],
            #     # "transport_cost":  lambda m: m.step_data["transport_cost"],
            #     # "maintenance":     lambda m: m.step_data["maintenance"],
            #     # "newcomer_bid":    lambda m: m.step_data["newcomer_bid"],
        }

        agent_reporters      = {
            "time_step":             lambda a: a.model.schedule.time,
            # # "agent_class":       lambda a: type(a),
            "agent_type":            lambda a: type(a).__name__,
            "bidder_name":           lambda a: getattr(a, "bidder_name", None)           if isinstance(a, Bid_Storage) else None,
            "bidder_savings":        lambda a: getattr(a, "bidder_savings", None)        if isinstance(a, Bid_Storage) else None,
            "distance":              lambda a: getattr(a, "distance_from_center", None)  if isinstance(a, Bid_Storage) else None,
            "transport_cost":        lambda a: getattr(a, "transport_cost", None)        if isinstance(a, Bid_Storage) else None,
            "bid":                   lambda a: getattr(a, "bid_value", None)             if isinstance(a, Bid_Storage) else None,
            "R_N":                   lambda a: getattr(a, "R_N", None)                   if isinstance(a, Bid_Storage) else None,
            # "Density":               lambda a: getattr(a, "density", None)               if isinstance(a, Bid_Storage) else None,
            # # "id":                lambda a: a.unique_id,
            # "x":                 lambda a: a.pos[0],
            # "y":                 lambda a: a.pos[1],
            # "is_working":        lambda a: None if not isinstance(a, Person) else 1 if a.unique_id in a.model.workforce.workers else 0,  # TODO does this need to be in model? e.g. a.model.workforce
            # "is_working_check":  lambda a: None if not isinstance(a, Person) else a.is_working_check,
            # "working_period":    lambda a: getattr(a, "working_period", None)  if isinstance(a, Person)       else None,
            # "wage_delta":        lambda a: getattr(a, "wage_delta", None)      if isinstance(a, Firm)         else None,
            # "p_dot":             lambda a: getattr(a, "p_dot", None)           if isinstance(a, Land)         else None,
            # "net_rent":          lambda a: getattr(a, "net_rent", None)        if isinstance(a, Land)         else None,
            # "warranted_rent":    lambda a: getattr(a, "warranted_rent", None)  if isinstance(a, Land)         else None,
            # "warranted_price":   lambda a: getattr(a, "warranted_price", None) if isinstance(a, Land)         else None,
            # "realized_price":    lambda a: getattr(a, "realized_price", None)  if isinstance(a, Land)         else None,
            # "realized_all_steps_price": lambda a: getattr(a, "realized_all_steps_price", None)  if isinstance(a, Land) else None,
            # "sold_this_step":    lambda a: getattr(a, "sold_this_step", None)  if isinstance(a, Land)         else None,
            # "ownership_type":    lambda a: getattr(a, "ownership_type", None)  if isinstance(a, Land)         else None,
            # "distance_from_center": lambda a: getattr(a, "distance_from_center", None) if isinstance(a, Land) else None,
        }

        

        self.datacollector  = DataCollector(model_reporters = model_reporters,
                                            agent_reporters = agent_reporters
                                            )

    def record_run_data_to_file(self):
        model_out = self.datacollector.get_model_vars_dataframe()
        agent_out = self.datacollector.get_agent_vars_dataframe()

        # Save agent data
        if agent_out is not None:
            try:
                agent_out.to_csv(self.agent_filepath, index=False)
            except Exception as e:
                logging.error("Error saving agent data: %s", str(e))

        # Save model data
        if model_out is not None:
            try:
                model_out.to_csv(self.model_filepath, index=False)
            except Exception as e:
                logging.error("Error saving model data: %s", str(e))

    def reset_step_data_lists(self):
        # Reset all lists within step_data
        for key in self.step_data:
            self.step_data[key] = []
