import logging
import os
import yaml
# import functools
import datetime
import random
import string
from typing import Dict, List
# from contextlib import contextmanager
import subprocess
# import math
# import numpy as np
# import pandas as pd
from scipy.spatial import distance

from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

from model.agents import Land, Person, Firm, Investor, Bank, Realtor
from model.schedule import RandomActivationByBreed

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
            'r_investor': 0.05,            # Next best alternative return for investor
            'property_tax_rate': 0.04,     # tau, annual rate, was c
            'housing_services_share': 0.3, # a
            'maintenance_share': 0.2,      # b
            'max_mortgage_share': 0.9,
            'ability_to_carry_mortgage': 0.28,
            'wealth_sensitivity': 0.1,
            'cg_tax_per':   0.01, # share 0-1
            'cg_tax_invest': 0.15, # share 0-1
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
        self.time_step = 0

        # self.setup_run_data_collection()

        # logging.basicConfig(filename=self.log_filename,
        #             filemode='w',
        #             level=logging.DEBUG,
        #             format='%(asctime)s %(name)s %(levelname)s:%(message)s')
        # self.logger = logging.getLogger(__name__)

        # logging.getLogger('matplotlib').setLevel(logging.ERROR) 

        
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
        min_savings = -20000
        max_savings = 200000
        no_steps = 5
        step_size = (max_savings - min_savings) / (no_steps - 1)
        self.newcomer_savings = [min_savings + i * step_size for i in range(no_steps)]
        # print(f'Newcomer savings: {self.newcomer_savings}')

        self.newcomer_bid_rent_history = []
        
        # self.setup_mesa_data_collection()
        # self.record_step_data()

        # # Run the firm for several steps to stabilize
        # for i in range(2):
        #     # People check if it's worthwhile to work
        #     self.schedule.step_breed(Person, step_name='work_if_worthwhile_to_work')

        #     # Firms update wages
        #     self.schedule.step_breed(Firm)   

    def step_fast(self):
        self.time_step += 1

        # Firm updates wages based on agglomeration population
        self.firm.step()

        # Firm updates agglomeration population based on calculated city extent
        extent = self.city_extent_calc
        self.firm.N = self.firm.get_N_from_city_extent(extent)

        # Calculate bid_rent values function of distance and person's savings
        # TODO does this exclude some of the city, effectively rounding down? Do rounding effects matter for the city extent/population calculations?
        # TODO could speed up by making more sparse
        dist = 0
        newcomer_bid_rent_values = []
        while dist <= extent:
            # TODO calculate the bid rent for the investor
            # investor_bid_rent = dist
            for savings_value in self.newcomer_savings:
                # print(savings_value)
                # TODO calculate the bid rent for the investor
                newcomer_bid_rent = savings_value # will make this a function - STORE FOR NOW
                newcomer_bid_rent_values.append(newcomer_bid_rent)
            self.newcomer_bid_rent_history.append(newcomer_bid_rent_values)
            dist += 1
            # print(f'Distance: {dist}')
        # TODO store the grid of output data

        # Store data about relationship between investor and person bid rent curves
