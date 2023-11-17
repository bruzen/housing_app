import math
import logging
import numpy as np
from typing import Union
from collections import defaultdict
from scipy.spatial import distance

from mesa import Agent

logging.basicConfig(filename='logfile.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

class Land(Agent):
    """Land parcel.

    :param unique_id: An integer identifier.
    :param model: The main city model.
    :param pos: The land parcel's location on the spatial grid.
    :param resident: The agent who resides at this land parcel.
    :param owner: The agent who owns this land parcel.
    """

    @property 
    def market_rent(self):
        return self.warranted_rent

    @property
    def net_rent(self):
        return self.warranted_rent - self.maintenance - self.property_tax # TODO check we don't use net rent for warranted_price
    
    @property
    def appraised_price(self):
        return self.warranted_price

    @property
    def property_tax(self):
        tau              = self.property_tax_rate
        appraised_price  = self.appraised_price
        return tau * appraised_price

    @property
    def maintenance(self):
        a                 = self.model.housing_services_share
        b                 = self.model.maintenance_share
        subsistence_wage  = self.model.firm.subsistence_wage # subsistence_wage
        return a * b * subsistence_wage

    def __init__(self, unique_id, model, pos, 
                 property_tax_rate = 0., 
                 resident = None, owner = None):
        super().__init__(unique_id, model)
        self.pos                  = pos
        self.property_tax_rate    = property_tax_rate
        self.resident             = resident
        self.owner                = owner
        self.distance_from_center = self.calculate_distance_from_center()
        self.transport_cost           = self.calculate_transport_cost()
        self.warranted_rent           = self.get_warranted_rent()
        self.warranted_price          = self.get_warranted_price()
        self.realized_price           = - 1
        self.realized_all_steps_price = - 1
        self.ownership_type           = - 1

    def step(self):
        self.warranted_rent  = self.get_warranted_rent()
        self.warranted_price = self.get_warranted_price()

        # Prepare price data for the current step
        price_data = {
            'land_id': self.unique_id,
            'warranted_price': self.warranted_price,
            'time_step': self.model.time_step,
            'transport_cost': self.transport_cost,
            'wage': self.model.firm.wage
        }
        
        # Add the price data to the model's step price data
        self.model.step_price_data.append(price_data)

        # TODO Flip
        self.realized_price         = - 1 # Reset to show realized price in just this time_step
        # self.realized_all_steps_price = - 1 # 

        if self.resident is None:
            logger.warning(f'Land has no resident: {self.unique_id}, pos {self.pos}, resident {self.resident}, owner {self.owner}')

    def calculate_distance_from_center(self, method='euclidean'):
        if method == 'euclidean':
            return distance.euclidean(self.pos, self.model.center)
        elif method == 'cityblock':
            return distance.cityblock(self.pos, self.model.center)
        else:
            raise ValueError("Invalid distance calculation method."
                            "Supported methods are 'euclidean' and 'cityblock'.")

    def calculate_transport_cost(self):
        cost = self.distance_from_center * self.model.transport_cost_per_dist
        return cost

    def change_owner(self, new_owner, old_owner):
        if not self.check_owners_match:
            owner_properties = ' '.join(str(prop.unique_id) if hasattr(prop, 'unique_id') else str(prop) for prop in self.owner.properties_owned)
            logger.error(f'In change_owner, property owner does not match an owner in owners properties_owned list: property {self.unique_id}, property\'s owner {self.owner.unique_id}, owner\'s properties {owner_properties} ')

        if not self.owner == old_owner:
            logger.error(f'In change_owner, the old_owner must own a property in order to transfer ownership: property {self.unique_id}, old_owner {old_owner.unique_id}, property\'s owner {self.owner.unique_id}')

        self.owner = new_owner

        # Remove the land from the old owner's properties_owned list
        old_owner.properties_owned.remove(self)

        # Add the land to the new owner's properties_owned list
        new_owner.properties_owned.append(self)

        # # Update the owner type
        # if isinstance(self.owner, Person):
        #     self.ownership_type = 1 # 'Person'
        # elif isinstance(self.owner, Investor):
        #     self.ownership_type = 2 # 'Investor'
        # else:
        #     self.ownership_type = 3 # 'Other'
        #     logger.warning(f'Land {self.unique_id} owner not a person or investor. Owner: {self.owner}')


    def get_warranted_rent(self):
        wage_premium     = self.model.firm.wage_premium
        subsistence_wage = self.model.firm.subsistence_wage
        a                = self.model.housing_services_share
        # return max(wage_premium - self.transport_cost, 0)
        # Note, this is the locational_rent. It is the warranted level of extraction. It is the economic_rent when its extracted.
        # We set the rural land value to zero to study the urban land market, with the agricultural price, warranted rent would be:
        return max(wage_premium - self.transport_cost + a * subsistence_wage, 0)
        # TODO add amenity + A
        # TODO should it be positive outside the city? How to handle markets outside the city if it is?

    def get_warranted_price(self):
        return self.warranted_rent / self.model.r_prime

    def check_owners_match(self):
        for owned_property in self.owner.properties_owned:
            if self.unique_id == owned_property.unique_id:
                return True
        return False
 
    def __str__(self):
        return f'Land {self.unique_id} (Dist. {self.distance_from_center}, Pw {self.warranted_price})'

class Person(Agent):
    @property
    def borrowing_rate(self):
        """Borrowing rate of the person.

        Returns:
        The borrowing rate calculated based on the model's  \
        target rate and individual wealth adjustment.
        """
        return self.model.r_target + self.individual_wealth_adjustment

    @property
    def individual_wealth_adjustment(self):
        """Individual wealth adjustment. Added on to the agent's mortgage 
        borrowing rate. It depends on the agent's wealth.

        # TODO: Fix
 
        Formula for interest rate they get: r_target + K/(W-W_min) - K/(W_mean-W_min)
        Formula for adjustment: K/(W-W_min) - K/(W_mean-W_min)
        K is wealth sensitivity parameter

        Returns:
        The individual wealth adjustment value.
        """
        # r_target = self.model.r_target
        # K        = self.model.wealth_sensitivity
        # W        = self.get_wealth() 
        # W_min
        # W_mean
        return 0.002

    def __init__(self, unique_id, model, pos, init_working_period = 0,
                 savings = 0., debt = 0.,
                 residence_owned = None):
        super().__init__(unique_id, model)
        self.pos = pos
        self.workforce = self.model.workforce

        self.init_working_period = init_working_period
        self.working_period      = init_working_period
        self.savings             = savings

        self.properties_owned    = []
        self.residence           = residence_owned

        self.bank                = self.model.bank 
        self.amenity             = 0.

        # If the agent initially owns a property, set residence and owners
        if self.residence:
            self.properties_owned.append(self.residence)
            if self.residence.owner is not None:
                logger.warning(f'Overrode property {self.residence.unique_id} owner {self.residence.owner} in init. Property now owned by {self.unique_id}.')
            self.residence.owner = self

            if self.residence.resident is not None:
                logger.warning(f'Overrode property {self.residence.unique_id} resident {self.residence.resident} in init. Property resident now {self.unique_id}.')
            self.residence.resident = self

        # Count time step and track whether agent is working
        self.count               = 0 # TODO check if we still use this
        self.is_working_check    = 0 # TODO delete?

        # else:
        #     self.workforce.remove(self, self.workforce.workers)

    def step(self):
        self.count              += 1
        self.working_period     += 1
        premium                  = self.model.firm.wage_premium

        # Non-residents
        if not isinstance (self.residence, Land):
            # Newcomers who don't find a home leave the city
            if (self.unique_id in self.workforce.newcomers):
                # if (self.residence == None):
                if self.count > 0:
                    logger.debug(f'Newcomer removed {self.unique_id}') #  removed, who owns {self.properties_owned}, count {self.count}')
                    self.remove()
                    return  # Stop execution of the step function after removal
            # Everyone who is not a newcomer leaves if they have no residence
            else:
                logger.warning(f'Non-newcomer with no residence removed: {self.unique_id}, count {self.count}')
                self.remove()
                return  # Stop execution of the step function after removal

        # Urban workers
        elif premium > self.residence.transport_cost:
            self.workforce.add(self, self.workforce.workers)
            # Agents list properties a step before they stop working for consistent worker numbers
            if self.working_period >= self.model.working_periods:
                if self.residence in self.properties_owned:
                    self.workforce.add(self, self.workforce.retiring_urban_owner)
                    listing = Listing(self, self.residence)
                    self.model.realtor.bids[listing] = []
                    # TODO Contact bank. Decide: sell, rent or keep empty
                    logger.debug(f'Agent is retiring: {self.unique_id}, period {self.working_period}')

            if self.working_period > self.model.working_periods:
                if self.residence in self.properties_owned:
                    self.workforce.remove(self, self.workforce.workers)
                    logger.warning(f'Urban homeowner still in model: {self.unique_id}, working_period {self.working_period}')
                else:
                    self.working_period = 1
                    self.savings        = 0
                    # TODO what should reset savings/age be for new renters? This simply resets keeping the initial distribution of ages cycling outside the city

        # Rural population
        else:
            self.workforce.remove(self, self.workforce.workers)
            if self.working_period > self.model.working_periods:
                self.working_period = 1
                self.savings        = 0
                # TODO what should reset savings/age be? This simply resets keeping the initial distribution of ages cycling outside the city
                
        # Update savings and wealth
        self.savings += self.model.savings_per_step # TODO debt, wealth
        # self.wealth  = self.get_wealth()

        # TODo consider doing additional check for retiring agents who are still in model
        # if self.unique_id in self.workforce.retiring_urban_owner:        
        #     if (self.residence):
        #         if premium > self.residence.transport_cost:
        #             logger.warning(f'Removed retiring_urban_owner agent {self.unique_id} in step, working, with residence, properties owned {len(self.properties_owned)} which was still in model.')
        #             self.remove()
        #         else:
        #             logger.warning(f'retiring_urban_owner agent {self.unique_id}, not working, with residence, still in model.')
        #     else:
        #         logger.warning(f'retiring_urban_owner agent {self.unique_id}, witout residence, still in model.')
        # else:
        #     logger.debug(f'Agent {self.unique_id} has no residence.')

        # TODO TEMP? use is_working_check to perform any checks
        if self.residence:
            if premium > self.residence.transport_cost:
                self.is_working_check = 1
            else:
                self.is_working_check = 0

            # TODO Temp
            if isinstance(self.residence.owner, Person):
                self.residence.ownership_type = 1 # 'Person'
            elif isinstance(self.residence.owner, Investor):
                self.residence.ownership_type = 2 # 'Investor'
            else:
                self.residence.ownership_type = 3 #'Other'


    def bid(self):
        """Newcomers bid on properties for use or investment value."""
        
        # logger.debug(f'Newcomer bids: {self.unique_id}, count {self.count}')

        # W = self.savings # TODO fix self.get_wealth()
        S = self.savings
        r = self.borrowing_rate
        r_prime  = self.model.r_prime
        r_target = self.model.r_target # TODO this is personal but uses same as bank. Clarify.        
        wage     = self.model.firm.wage
        standard_mortgage_share = 0.8 # TODO make this a parameter
        # average_wealth = # TODO? put in calculation

        # Max mortgage
        M = 0.28 * (wage + r * S) / r_prime

        # Max mortgage share
        m = 0.8 # TODO get mortgage share
         # m = max_mortgage_share - lenders_wealth_sensitivity * average_wealth / W

        for listing in self.model.realtor.bids:

            # First Calculate value of purchase (max bid)
            R_N      = listing.sale_property.net_rent # Need net rent for P_bid
            bid_type = 'value_limited'
            P_bid    = self.model.bank.get_max_bid(R_N, r, r_target, m, listing.sale_property.transport_cost)
            # logger.warning(f'Max bid: {self.unique_id}, bid {P_bid}, R_N {R_N}, r {r}, r {r_target}, m {m}, transport_cost {listing.sale_property.transport_cost}')

            if S/(1-m) <= P_bid:
                bid_type = 'equity_limited'
                P_bid = S/(1-m)
                # logger.warning(f'Newcomer bids EQUITY LIMITED: {self.unique_id}, bid {P_bid}') #, S {S}, m {m}, .. ')

            if (0.28 * (wage + r * S) / r_prime)  <= P_bid:
                bid_type = 'income_limited'
                P_bid = 0.28 * (wage + r * S) / r_prime
                logger.warning(f'Newcomer bids INCOME LIMITED: {self.unique_id}, bid {P_bid}')

            if P_bid > 0:
                # logger.warning(f'Newcomer bids: {self.unique_id}, bid {P_bid}')
                self.model.realtor.add_bid(self, listing, P_bid, bid_type)

            else:
                pass
                # logger.warning(f'Newcomer doesnt bid: {self.unique_id}, bid {P_bid}, sale_property {listing.sale_property.unique_id}')

            # # Old logic, replaced by version above
            # # Max desired bid
            # R_N = listing.sale_property.net_rent
            # # P_max_bid = self.model.bank.get_max_bid(R_N, r, r_target, m, listing.sale_property.transport_cost)

            # mortgage_share_max = m * P_max_bid # TODO this should have S in it. 
            # mortgage_total_max = M

            # # Agents cannot exceed any of their constraints
            # if mortgage_share_max < mortgage_total_max:
            #     # Mortgage share limited
            #     if mortgage_share_max + S < P_max_bid:
            #         P_bid = mortgage_share_max + S
            #         bid_type = 'mortgage_share_limited'
            #     # Max bid limited
            #     else:
            #         P_bid = P_max_bid
            #         bid_type = 'max_bid_limited'
            #     mortgage = P_bid - S # TODO is this right? what about savings. 
            # else:
            #     mortgage = M
            #     # Mortgage total limited
            #     if mortgage_total_max + S < P_max_bid:
            #         P_bid = mortgage_total_max + S
            #         bid_type = 'mortgage_total_limited'
            #     # Max bid limited
            #     else:
            #         P_bid = P_max_bid
            #         bid_type = 'max_bid_limited'

    def get_wealth(self):
        # TODO Wealth is properties owned, minus mortgages owed, plus savings.
        # W = self.resident_savings + self.P_expected - self.M
        return self.savings

    def remove(self):
        self.model.removed_agents += 1
        self.workforce.remove_from_all(self)
        # self.model.grid.remove(self)
        self.model.schedule.remove(self)
        # x, y = self.pos
        # self.model.grid.remove_agent(x,y,self)
        self.model.grid.remove_agent(self)
        # logger.debug(f'Person {self.unique_id} removed from model')
        # TODO If agent owns property, get rid of property

    def __str__(self):
        return f'Person {self.unique_id}'

class Firm(Agent):
    """Firm.

    :param unique_id: An integer identifier.
    :param model: The main city model.
    :param pos: The firms's location on the spatial grid.
    :param init_wage_premium: initial urban wage premium.
    """

    # # TODO include seed population?
    # @property
    # def N(self):
    #     """total_no_workers"""
    #     total_no_workers = self.model.workforce.get_agent_count(self.model.workforce.workers)
    #     return total_no_workers * self.density + self.seed_population

    def __init__(self, unique_id, model, pos, 
                 subsistence_wage,
                 init_wage_premium_ratio,
                 alpha, beta, gamma,
                 price_of_output, r_prime,
                #  wage_adjustment_parameter,
                #  firm_adjustment_parameter,
                 seed_population,
                 density,
                 A,
                 overhead,
                 mult,
                 c,
                 adjN,
                 adjk,
                 adjn,
                 adjF,
                 adjw,
                 dist,
                 init_agglomeration_population,
                 init_F,
                 init_k,
                 init_n,
                 ):
        super().__init__(unique_id, model)
        self.pos             = pos

        # Old initialization calculations
        # # Calculate scale factor A for a typical urban firm
        # Y_R      = n_R * subsistence_wage / beta_F
        # Y_U      = self.n * self.wage / beta_F
        # k_R      = alpha_F * Y_R / self.r
        # self.k   = alpha_F * Y_U / self.r
        # self.A_F = 3500 # Y_R/(k_R**alpha_F * n_R * self.subsistence_wage**beta_F)

        # TEMP New parameter values
        self.subsistence_wage = subsistence_wage # subsistence_wage
        self.alpha    = alpha
        self.beta     = beta
        self.gamma    = gamma
        self.price_of_output  = price_of_output
        self.seed_population  = seed_population
        self.density  = density
        self.A        = A
        self.overhead = overhead    # labour overhead costs for firm
        self.mult     = mult
        self.c        = c
        self.adjN     = adjN
        self.adjk     = adjk
        self.adjn     = adjn
        self.adjF     = adjF
        self.adjw     = adjw
        self.dist     = dist
        # agent_count = 50 # TODO comes from agents deciding
        self.r        = r_prime # Firm cost of capital

        # Initial values # TODO do we need all these initial values?
        self.y        = 100000
        self.Y        = 0
        self.F        = init_F
        self.k        = init_k #1.360878e+09 #100
        self.n        = init_n
        self.F_target = init_F
        #self.k_target = 10000
        #self.n_target = 100        
        #self.y_target = 10000
        self.N = self.F * self.n
        self.agglomeration_population = init_agglomeration_population # population TODO change this will be confused with price
        self.wage_premium = init_wage_premium_ratio * self.subsistence_wage 
        self.wage         = self.wage_premium + self.subsistence_wage
        self.MPL          = self.beta  * self.y / self.n  # marginal value product of labour known to firms

    def step(self):
        # GET POPULATION AND OUTPUT TODO replace N with agent count
        self.N = self.get_N() # TODO make sure all relevant populations are tracked - n, N, N adjustedx4/not, agent count, agglomeration_population
        self.agglomeration_population = self.mult * self.N + self.seed_population
        self.n =  self.N / self.F # distribute workforce across firms
        self.y = self.A * self.agglomeration_population**self.gamma *  self.k**self.alpha * self.n**self.beta

        # ADJUST WAGE
        self.MPL = self.beta  * self.y / self.n  # marginal value product of labour known to firms
        self.wage_target = self.subsistence_wage + (self.MPL - self.subsistence_wage) / (1 + self.overhead)       #self.wage_target = self.MPL / (1 + self.overhead) # (1+self.overhead) # economic rationality implies intention
        self.wage = (1 - self.adjw) * self.wage + self.adjw * self.wage_target # assume a partial adjustment process
        
        # FIND POPULATION AT NEW WAGE
        self.wage_premium = self.wage - self.subsistence_wage # find wage available for transportation
        #self.dist = self.wage_premium / self.c  # find calculated extent of city at wage
        #self.N = self.dist * self.model.height * self.density / self.mult # calculate total firm population from city size # TODO make this expected pop
        #self.n =  self.N / self.F # distribute workforce across firms

        # ADJUST NUMBER OF FIRMS
        self.F_target = self.F * self.wage_target/self.wage  # this is completely arbitrary but harmless
        self.F = (1 - self.adjF) * self.F + self.adjF * self.F_target
 
        # ADJUST CAPITAL STOCK 
        self.y_target = self.price_of_output * self.A * self.agglomeration_population**self.gamma *  self.k**self.alpha * self.n**self.beta
        self.k_target = self.alpha * self.y_target/self.r
        self.k = (1 - self.adjk) * self.k + self.adjk * self.k_target
    
        #self.F_target = self.F * self.n_target/self.n  #this is completely argbitrary but harmless
        # self.F_target = self.F*(self.n_target/self.n)**.5 # TODO name the .5
        ####self.F_target = (1-self.adjF)*self.F + self.adjF*self.F*(self.n_target/self.n) 
        #self.N_target = self.F_target * self.n_target
        #self.N = (1 - self.adjN) * self.N + self.adjN * self.N_target
        #####self.N = self.N*1.02
        #self.F = (1 - self.adjF) * self.F + self.adjF * self.F_target
        #self.k = (1 - self.adjk) * self.k + self.adjk * self.k_target
        # n = N/F 
        #self.wage_premium = self.c * math.sqrt(self.mult * self.N / (2 * self.density)) # TODO check role of multiplier
        #self.wage = self.wage_premium + self.subsistence_wage

        # # TODO Old firm implementation. Cut.
        # # Calculate wage, capital, and firm count given number of urban workers
        # self.n = self.N/self.F
        # self.y = self.output(self.N, self.k, self.n)

        # self.n_target = self.beta_F * self.y / self.wage
        # self.y_target = self.output(self.N, self.k, self.n_target)
        # self.k_target = self.alpha_F * self.y_target / self.r

        # # N_target_exist = n_target/self.n * self.N
        # adj_f = self.firm_adjustment_parameter # TODO repeats
        # self.F_target = self.n_target/self.n * self.F
        # self.F_next = (1 - adj_f) * self.F + adj_f * self.F_target
        # self.N_target_total = self.F_next * self.n_target
        # self.F_next_total = self.N_target_total / self.n_target

        # # adj_l = 1.25 # TODO self.labor_adjustment_parameter
        # # N_target_total = adj_l * n_target/self.n * self.N
        # # N_target_new = n_target * self.Z * (MPL - self.wage)/self.wage * self.F # TODO - CHECK IS THIS F-NEXT?

        # c = self.model.transport_cost_per_dist
        # self.wage_premium_target = c * math.sqrt(self.N_target_total/(2*self.density))        

        # k_next = self.k_target # TODO fix

        # adj_w = self.wage_adjustment_parameter
        # # self.wage_premium = self.wage_premium_target # TODO add back in wage adjusment process
        # # self.wage_premium = (1-adj_w) * self.wage_premium + adj_w * self.wage_premium_target
        # if self.model.time_step < 3:
        #     self.wage_premium = (1-adj_w)*self.wage_premium + adj_w * self.wage_premium_target
        # else:
        #     self.wage_premium += 100
        # self.k = k_next
        # self.F = self.F_next_total # OR use F_total

    # def output(self, N, k, n):
    #     A_F     = self.A_F
    #     alpha_F = self.alpha_F
    #     beta_F  = self.beta_F
    #     gamma   = self.model.gamma
    #     return A_F * N**gamma * k**alpha_F * n**beta_F

    def get_N(self):
        # If the city is in the bottom corner center_city is false, and effective population must be multiplied by 4
        # TODO think about whether this multiplier needs to come in elsewhere
        worker_agent_count = self.model.workforce.get_agent_count(self.model.workforce.workers)
        if self.model.center_city:
            N = self.density * worker_agent_count
        else:
            N = 4 * self.density * worker_agent_count
        # TODO handle divide by zero errors
        if N == 0:
            N = 1
        return N

class Bank(Agent):
    def __init__(self, unique_id, model, pos,
                 r_prime = 0.05, max_mortgage_share = 0.9,
                 ):
        super().__init__(unique_id, model)
        self.pos = pos

    def get_max_bid(self, R_N, r, r_target, m, transport_cost):
        T      = self.model.mortgage_period
        delta  = self.model.delta
        p_dot  = self.model.p_dot #(transport_cost)

        if R_N is not None and r is not None and r_target is not None and m is not None and p_dot is not None:
            R_NT   = ((1 + r)**T - 1) / r * R_N
            # return R_NT / ((1 - m) * r_target/(delta**T) - p_dot) 
            return R_NT / ((1 - m) * r_target/(delta**T) - p_dot +(1+r)**T*m) # Revised denominator from eqn 6:20

        else:
            logger.error(f'Get_max_bid None error Rn {R_N}, r {r}, r_target {r_target}, m {m}, p_dot {p_dot}')
            return 0. # TODO Temp

    def get_average_wealth(self):
        rural_home_value     = self.get_rural_home_value()
        avg_locational_value = self.model.firm.wage_premium / (3 * self.model.r_prime)
        return rural_home_value + avg_locational_value
        # AVERAGE_WEALTH_CALCULATION
        # The value of average_wealth.
        # # value of a home + savings half way through a lifespan.
        # # Value of house on average in the city - know the area and volume of a cone. Cone has weight omega, the wage_premium
        # avg_wealth = rural_home_value + avg_locational_value + modifier_for_other_cities_or_capital_derived_wealth
        # where:
        # avg_locational_value = omega / (3 * r_prime)
        # TODO consider adding modifier_for_other_cities_or_capital_derived_wealth
        # TODO check if we need to adjust if not center_city

    def get_rural_home_value(self):
        # r is the bank rate since this is the bank's assessment of value. a is the housing share, and a * subsistence_wage is the value of the housing services since we've fixed the subsistence wage and all houses are the same.
        a                = self.model.housing_services_share
        subsistence_wage = self.model.firm.subsistence_wage
        r                = self.model.r_prime
        return a * subsistence_wage / r

class Investor(Agent):

    # @property
    # def borrowing_rate(self):
    #     self.model.r_target
    
    def __init__(self, unique_id, model, pos, properties_owned = []):
        super().__init__(unique_id, model)
        self.pos = pos
        self.borrowing_rate = self.model.r_target

        # Properties for bank as an asset holder
        # self.property_management_costs = property_management_costs # TODO 
        self.properties_owned      = properties_owned

    def bid(self):
        # """Investors bid on investment properties."""
        m = 0.9 # TODO fix
        r = self.borrowing_rate
        r_target = self.model.r_target

        for listing in self.model.realtor.bids:
            R_N      = listing.sale_property.net_rent
            P_bid    = self.model.bank.get_max_bid(R_N, r, r_target, m, listing.sale_property.transport_cost)
            bid_type = 'investor'
            # mortgage = m * P_bid
            logger.debug(f'Bank {self.unique_id} bids {P_bid} for property {listing.sale_property.unique_id}, if val is positive.')
            if P_bid > 0:
                self.model.realtor.add_bid(self, listing, P_bid, bid_type)
            else:
                logger.debug(f'Investor doesn\'t bid: {self.unique_id}')

    def __str__(self):
        return f'Investor {self.unique_id}'

class Realtor(Agent):
    """Realtor agents connect sellers, buyers, and renters."""
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.workforce = self.model.workforce

        # self.sale_listings = []
        self.rental_listings = []

        self.bids = defaultdict(list)

    def step(self):
        pass

    def add_bid(self, bidder, listing, price, bid_type= ''):
        # Type check for bidder and property
        if not isinstance(bidder, (Person, Investor)):
            logger.error(f'Bidder in add_bid {bidder.unique_id} is not a Person or Investor. {bidder}')
        if not isinstance(listing, Listing):
            logger.error(f'Listing in add_bid is not a Listing instance.')
        if not isinstance(price, (int, float)):
            logger.error(f'Price in add_bid must be a numeric value (int or float).')
        if not isinstance(bid_type, (str)):
            logger.error(f'Bid type in add_bid must be a string.')
    
        bid = Bid(bidder, price, bid_type)
        self.bids[listing].append(bid)

    def sell_homes(self):
        # Allocate properties based on bids
        allocations = []

        logger.debug(f'Number of listed properties to sell: {len(self.bids)}')
        for listing, property_bids in self.bids.items():
            # # Look at all bids for debugging
            # bid_info = [f'bidder {bid.bidder} bid {bid.bid_price}' for bid in property_bids]
            # logger.debug(f'Listed property: {listing.sale_property.unique_id} has bids: {len(property_bids)}, {", ".join(bid_info)}')

            if property_bids:
                property_bids.sort(key=lambda x: x.bid_price, reverse=True)
                highest_bid              = property_bids[0]
                highest_bid_price        = property_bids[0].bid_price
                second_highest_bid_price = property_bids[1].bid_price if len(property_bids) > 1 else 0

                final_price = highest_bid_price # TODO decide highest price here or elsewhere?

                allocation = Allocation(
                                        buyer                    = highest_bid.bidder,
                                        seller                   = listing.seller,
                                        sale_property            = listing.sale_property,
                                        final_price              = final_price,
                                        highest_bid_price        = highest_bid.bid_price,
                                        second_highest_bid_price = second_highest_bid_price,
                                        )
                allocations.append(allocation)
            # TODO *** compute final price given wtp
            # TODO what if property doesn't sell? Is it relisted by original owner? Is the price different?

        # Complete transaction and clear existing bids
        self.complete_transactions(allocations)
        self.bids.clear()
        # self.sale_listings.clear() # TODO check do we clear listings here?
        return allocations # TODO returning for testing. Do we need this? Does it interfere with main code?

    def complete_transactions(self, allocations):
        for allocation in allocations:
            logger.debug(f'Property {allocation.sale_property.unique_id} sold by seller {allocation.seller} to {allocation.buyer} for {allocation.final_price}')
            # if not isinstance(allocation.seller, Person):
            #     logger.debug(f'Seller not a Person in complete_transaction: Seller {allocation.seller.unique_id}')

            # Record data for data_collection
            allocation.sale_property.realized_price = allocation.final_price
            allocation.sale_property.realized_all_steps_price = allocation.final_price # Record that it sold

            # Record data for forecasting
            new_row = {
            'land_id':        allocation.sale_property.unique_id,
            'realized_price': allocation.final_price,
            'time_step':      self.model.time_step,
            'transport_cost': allocation.sale_property.transport_cost,
            'wage':           self.model.firm.wage,
            }
            self.model.realized_price_data = self.model.realized_price_data.append(new_row, ignore_index=True)

            # if isinstance(allocation.seller, Person):
            #     pass
            #     # self.handle_seller_departure(allocation)
            # elif isinstance(allocation.seller, Investor):
            #     logger.debug(f'In complete_transaction, before purchase, seller is Investor, id {allocation.seller.unique_id}.')
            # else:
            #     logger.debug(f'In complete_transaction, before purchase, seller {allocation.seller.unique_id} was not a person or investor. Seller {allocation.seller}.')

            # logger.debug(f'Time {self.model.time_step}, Property {allocation.property.unique_id}, Price {allocation.property.realized_price}')
            if isinstance(allocation.buyer, Investor):
                self.handle_investor_purchase(allocation)
            elif isinstance(allocation.buyer, Person):
                self.handle_person_purchase(allocation)
            else:
                logger.warning('In complete_transaction, buyer was neither a person nor an investor.')

            self.handle_seller_departure(allocation)
            # if isinstance(allocation.seller, Person):
            #     logger.debug(f'Seller departure {allocation.seller}')
                
            # elif isinstance(allocation.seller, Investor):
            #     logger.debug(f'In complete_transaction, after purchase, seller is Investor, id {allocation.seller.unique_id}.')
            # else:
            #     logger.debug(f'In complete_transaction, after purchase, seller {allocation.seller.unique_id} was not a person or investor. Seller {allocation.seller}.')                

    def handle_investor_purchase(self, allocation):
        """Handles the purchase of a property by an investor."""
        allocation.sale_property.change_owner(allocation.buyer, allocation.seller)
        allocation.sale_property.resident = None
        allocation.buyer.residence = None
        logger.debug('Property %s sold to investor.', allocation.sale_property.unique_id)
        self.rental_listings.append(allocation.sale_property)

    def handle_person_purchase(self, allocation):
        """Handles the purchase of a property by a person."""
        allocation.sale_property.change_owner(allocation.buyer, allocation.seller)
        # if not allocation.sale_property.check_residents_match:
        #     logger.error(f'Sale property residence doesn\'t match before transfer: seller {seller.unique_id}, buyer {buyer.unique_id}')
        allocation.sale_property.resident = allocation.buyer
        allocation.buyer.residence = allocation.sale_property
        self.model.grid.move_agent(allocation.buyer, allocation.sale_property.pos)
        # if not allocation.sale_property.check_residents_match:
        #     logger.error(f'Sale property residence doesn\'t match after transfer: seller {seller.unique_id}, buyer {buyer.unique_id}')
        logger.debug('Property %s sold to newcomer %s.', allocation.sale_property.unique_id, allocation.buyer.unique_id)

        # if self.residence: # Already checked since residence assigned
        if self.model.firm.wage_premium > allocation.sale_property.transport_cost:
            self.workforce.add(allocation.buyer, self.workforce.workers)
            logger.debug(f'Person purchase: add person to workforce')
        else:
            logger.debug(f'Person purchase: don\'t add person to workforce')

        if allocation.buyer.unique_id in self.workforce.newcomers:
            logger.debug(f'Removing newcomer {allocation.buyer.unique_id}')
            self.workforce.remove(allocation.buyer, self.workforce.newcomers)
        else:
            logger.warning(f'Person buyer was not a newcomer: {allocation.buyer.unique_id}')
        # logger.debug(f'Time {self.model.time_step} New worker {buyer.unique_id} Loc {sale_property}') # TEMP

    def handle_seller_departure(self, allocation):
        """Handles the departure of a selling agent."""
        if isinstance(allocation.seller, Person):
            if allocation.seller.unique_id in self.workforce.retiring_urban_owner:
                logger.debug(f'Removing seller {self.unique_id}')
                allocation.seller.remove()
            else:
                logger.warning(f'Seller a Person but not retiring, so not removed: Seller {allocation.seller.unique_id}')
        elif isinstance(allocation.seller, Investor):
            logger.warning(f'Seller an Investor in handle_seller_departure: Seller {allocation.seller.unique_id}')
        else:
            logger.warning(f'Seller not an Investor or a Person in handle_seller_departure: Seller {allocation.seller.unique_id}')

    def rent_homes(self):
        """Rent homes listed by investors to newcomers."""
        logger.debug(f'{len(self.rental_listings)} properties to rent.')
        for rental in self.rental_listings:
            renter = self.model.create_newcomer(rental.pos)
            rental.resident = renter
            renter.residence = rental
            self.workforce.remove(renter, self.workforce.newcomers)
            logger.debug(f'Newly created renter {renter.unique_id} lives at '
                         f'property {renter.residence.unique_id} which has '
                         f'resident {rental.resident.unique_id}, owner {rental.owner.unique_id}, pos {renter.pos}.')
        self.rental_listings.clear()

class Listing:
    def __init__(
        self, 
        seller: Union[Person, Investor],
        sale_property: Land, 
        list_price: Union[float, int] = 0.0,
    ):
        if not isinstance(seller, (Person, Investor)):
            logger.error(f'Bidder in Listing {seller.unique_id} is not a Person or Investor, {seller}')
        if not isinstance(sale_property, Land):
            logger.error(f'sale_property in Listing {sale_property.unique_id} is not Land, {sale_property}')
        if not isinstance(list_price, (float, int)):
            logger.error(f'list_price in Listing must be a numeric value.')
               
        self.seller = seller
        self.sale_property = sale_property
        self.list_price = list_price

    def __str__(self):
        return f'Seller: {self.seller}, Property: {self.sale_property}, List Price: {self.list_price}'

class Bid:
    def __init__(
        self, 
        bidder: Union[Person, Investor],
        bid_price: Union[float, int],
        bid_type: str = '',
    ):
        if not isinstance(bidder, (Person, Investor)):
            logger.error(f'Bidder in Bid {bidder.unique_id} is not a Person or Investor, {bidder}')
        if not isinstance(bid_price, (float, int)):
            logger.error(f'Price in Bid must be a numeric value.')
        if not isinstance(bid_type, (str)):
            logger.error(f'Price in Bid must be a numeric value.')
               
        self.bidder = bidder
        self.bid_price = bid_price
        self.bid_type = bid_type

    def __str__(self):
        return f'Bidder: {self.bidder.unique_id}, Price: {self.price}, Type: {self.bid_type}'

class Allocation:
    def __init__(
        self, 
        buyer: Union[Person, Investor],
        seller: Union[Person, Investor],
        sale_property: Land,
        final_price: Union[float, int] = 0.0,
        highest_bid_price: Union[float, int] = 0.0,
        second_highest_bid_price: Union[float, int] = 0.0,
    ):
        if not isinstance(buyer, (Person, Investor)):
            logger.error(f'Successful buyer {buyer.unique_id} in Allocation must be of type Person or Investor.')
        if not isinstance(seller, (Person, Investor)):
            logger.error(f'Seller {seller.unique_id} in Allocation must be of type Person or Investor.')
        if not isinstance(sale_property, Land):
            logger.error(f'Property {property.unique_id} in Allocation must be of type Land.')
        if not isinstance(final_price, (float, int)):
            logger.error(f'Final price in Allocation must be a numeric value.')
        if not isinstance(highest_bid_price, (float, int)):
            logger.error(f'Highest bid in Allocation must be a numeric value.')
        if not isinstance(second_highest_bid_price, (float, int)):
            logger.error(f'Second highest bid in Allocation must be a numeric value.')

        self.buyer                    = buyer
        self.seller                   = seller
        self.sale_property            = sale_property
        self.final_price              = final_price
        self.highest_bid_price        = highest_bid_price
        self.second_highest_bid_price = second_highest_bid_price

    def __str__(self):
        return f'Buyer: {self.buyer}, Seller {self.seller.unique_id}, Property: {self.sale_property}, Final Price: {self.final_price}, Highest Bid: {self.highest_bid_price}, Second Highest Bid: {self.second_highest_bid_price}'
