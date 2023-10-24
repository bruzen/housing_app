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
    def warranted_rent(self):
        wage_premium     = self.model.firm.wage_premium
        subsistence_wage = self.model.firm.subsistence_wage
        a                = self.model.housing_services_share
        return wage_premium - self.transport_cost + a * subsistence_wage # TODO add amenity + A

    @property 
    def market_rent(self):
        return self.warranted_rent

    @property
    def net_rent(self):
        return self.warranted_rent - self.maintenance - self.property_tax

    @property
    def warranted_price(self):
        return self.warranted_rent / self.model.r_prime
    
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
        self.transport_cost       = self.calculate_transport_cost()
        self.person_vs_investor_owner = 0 # random.randint(0, 1) # random.choice([True, False])  # TODO check who owner is and update.
        self.realized_price       = - 1 # random.randint(1, 10000)
        # TODO want to make distance from center, warranted price, realized price.
        # self.owner_types = np.array(['Person', 'Investor', 'Bank', 'Other']) # TEMP - move to model? Have agent type list?
        self.owner_type = 'Other'

    def step(self):
        # Prepare price data for the current step
        price_data = {
            'id': self.unique_id,
            'warranted_price': self.warranted_price,
            'time_step': self.model.time_step,
            'transport_cost': self.transport_cost,
            'wage': self.model.firm.wage
       }
        # Add the price data to the model's step price data
        self.model.step_price_data.append(price_data)

        probabilities = [0.01, 0.99]

        # TODO random values should use model's random number generator
        self.person_vs_investor_owner = np.random.choice([np.random.randint(0, 1), -1], size=None, p=probabilities) # random.randint(0, 1) # random.choice([True, False])  # TODO check who owner is and update.
        self.realized_price           = -1 # np.random.choice([np.random.uniform(0, 600000), -1], size=None, p=probabilities) # random.randint(1, 10000)
        # TODO do something with old realized prixe

        # Generate a random index to select an owner type
        # random_index = np.random.randint(0, len(self.owner_types))
        # Get the random owner type
        # self.owner_type = self.owner_types[random_index]
        if isinstance(self.owner, Person):
            self.owner_type = 'Person'
        elif isinstance(self.owner, Investor):
            self.owner_type = 'Investor'
        else:
            self.owner_type = 'Other'

        # # TODO TEMP!
        # if np.random.random() < 0.2:
        #     self.owner_type = 'Investor'

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

    def __str__(self):
        return f"Land {self.unique_id} (Dist. {self.distance_from_center}, Pw {self.warranted_price})"

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
        borrowng rate. It depends on the agent's wealth.

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
                logger.warning(f'Property {self.residence.unique_id} has \
                                 owner {self.residence.owner}, now \
                                 owned by {self.unique_id} in init.')
            self.residence.owner = self

            if self.residence.resident is not None:
                logger.warning(f'Property {self.residence.unique_id} has \
                                 resident {self.residence.resident}, now \
                                 assigned to {self.unique_id} in init.')
            self.residence.resident = self

        # Count time step and track whether agent is working
        self.count               = 0

    def step(self):
        self.count              += 1
        self.working_period     += 1

        premium = self.model.firm.wage_premium
        # Newcomers, who don't find a home, leave the city
        if (self.unique_id in self.workforce.newcomers):
            if (self.residence == None):
                if (self.count > 0):
                    logger.debug(f'Newcomer {self.unique_id} removed, who \
                                   owns {self.properties_owned}.')
                    self.remove()
            else:
                logger.error(f'Newcomer {self.unique_id} has a \
                               residence {self.residence.unique_id}, \
                               but was not removed from newcomer list.')
        
        elif (self.residence) and (self.unique_id not in self.workforce.retiring):
            # If it is worthwhile work
            if premium > self.residence.transport_cost:
                # Retire if past retirement age
                if (self.working_period > self.model.working_periods):
                    self.workforce.add(self, self.workforce.retiring)
                    # List homes for sale
                    if (self.residence in self.properties_owned):
                        # TODO Contact bank. Decide: sell, rent or keep empty
                        self.model.realtor.sale_listing.append(self.residence)
                        # TODO if residence is not owned, renter moves out
                else:
                    # Add the person to the workforce's workers dictionary if not already present
                    self.workforce.add(self, self.workforce.workers)
            # If it is not worthwhile to work
            else:
                # Remove the person from the workforce's workers dictionary if present
                self.workforce.remove(self, self.workforce.workers)
                if (self.working_period > self.model.working_periods):
                    # Population age cycles to model an outside urban population
                    self.working_period = 1 # TODO test demoographics

            # Update savings
            self.savings += self.model.savings_per_step # TODO debt, wealth
            self.wealth  = self.get_wealth()
        elif self.unique_id in self.workforce.retiring:
            logger.debug(f'Retiring agent {self.unique_id} still in model.')
            print(f'Retiring agent {self.unique_id} still in model.')

        else:
            logger.debug(f'Agent {self.unique_id} has no residence.')
            print(f'Agent {self.unique_id} has no residence.')

    def bid(self):
        """Newcomers bid on properties for use or investment value."""
        
        # W = self.savings # TODO fix self.get_wealth() # TODO use wealth in mortgage share and borrowing rate
        S = self.savings
        r = self.borrowing_rate
        r_prime  = self.model.r_prime
        r_target = self.model.r_target # TODO this is personal but uses same as bank. Clarify.        
        wage     = self.model.firm.wage

        # Max mortgage
        M = 0.28 * (wage + r * S) / r_prime

        # Max mortgage share
        m = 0.8

        for sale_property in self.model.realtor.sale_listing:

            #First Calculate value of purchase (max bid)
            R_N      = sale_property.net_rent # Need net rent for P_bid
            P_bid    = self.model.bank.get_max_bid(R_N, r, r_target, m, sale_property.transport_cost)
            bid_type = 'value_limited'

            if S/(1-m) <= P_bid:
                bid_type = 'equity_limited'
                P_bid = S/(1-m)

            if (0.28 * (wage + r * S) / r_prime)  <= P_bid: # Note not elif - this replaces the above
                bid_type = 'income_limited'
                P_bid = 0.28 * (wage + r * S) / r_prime

            if P_bid > 0:
                self.model.realtor.add_bid(self, sale_property, P_bid, bid_type)

            # # Old logic, replaced by version above
            # # Max desired bid
            # R_N = sale_property.net_rent
            # # P_max_bid = self.model.bank.get_max_bid(R_N, r, r_target, m, sale_property.transport_cost)

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
        # TODO Wealth is properties owned, minuse mortgages owed, plus savings.
        return self.savings

    def remove(self):
        self.model.removed_agents += 1
        self.workforce.remove_from_all(self)
        # self.model.grid.remove(self)
        self.model.schedule.remove(self)
        # x, y = self.pos
        # self.model.grid.remove_agent(x,y,self)
        self.model.grid.remove_agent(self)   

    def __str__(self):
        return f"Person {self.unique_id}"

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
        p_dot  = self.model.get_p_dot() #(transport_cost)

        if R_N is not None and r is not None and r_target is not None and m is not None and p_dot is not None:
            R_NT   = ((1 + r)**T - 1) / r * R_N
            return R_NT / ((1 - m) * r_target/(delta**T) - p_dot)
        else:
            print(f'get_max_bid error Rn {R_N}, r {r}, r_target {r_target}, m {m}, p_dot {p_dot}')
            return 0. # TODO Temp

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

    # def step(self):
    #     self.bid()

    def bid(self):
        # """Investors bid on investment properties."""
        m = 0.9 # TODO fix
        r = self.borrowing_rate
        r_target = self.model.r_target

        for sale_property in self.model.realtor.sale_listing:
            R_N      = sale_property.net_rent
            P_bid    = self.model.bank.get_max_bid(R_N, r, r_target, m, sale_property.transport_cost)
            bid_type = 'investor'
            mortgage = m * P_bid
            logger.debug(f'Bank {self.unique_id} bids {P_bid} for \
                        property {sale_property.unique_id}, if val is positive.')
            if P_bid > 0:
                self.model.realtor.add_bid(self, sale_property, P_bid, bid_type, mortgage)

class Realtor(Agent):
    """Realtor agents connect sellers, buyers, and renters."""
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.workforce = self.model.workforce

        self.sale_listing = []
        self.rental_listing = []

        self.bids = defaultdict(list)

    def step(self):
        pass

    def add_bid(self, bidder, sale_property, price, bid_type= "", mortgage=0.):
        # Type check for bidder and property
        if not isinstance(bidder, (Person, Investor)):
            raise ValueError("Bidder must be of type Person or Investor.")
        if not isinstance(sale_property, Land):
            raise ValueError("Property must be of type Land.")
        if not isinstance(price, (int, float)):
            raise ValueError("Price must be a numeric value (int or float).")
        
        bid = Bid(bidder, sale_property, price, bid_type, mortgage)
        self.bids[sale_property].append(bid)

    def sell_homes(self):
        # TODO maybe if DEBUG?
        # for key, value in self.bids.items():
        #     print(f'Key: {key}')
        #     for bid in value:
        #         print(f'  {bid}')
        
        # Allocate proeprties based on bids
        allocations = []
        for sale_property in self.bids.keys():
            if not isinstance(sale_property, Land):
                raise TypeError(f"Property in self.bids.keys is not of type 'Land'.")

            property_bids = self.bids[sale_property]
            if len(property_bids) > 0:
                property_bids.sort(key=lambda x: x.price, reverse=True)
                highest_bid = property_bids[0]
                second_highest_bid = property_bids[1].price if len(property_bids) > 1 else 0
                final_price = highest_bid.price
                allocation = Allocation(highest_bid.bidder, sale_property.owner, sale_property, final_price, highest_bid.price, second_highest_bid)
                # print(allocation)
                allocations.append(allocation)
            # TODO *** compute final price given wtp

        # Complete transaction and clear existing bids
        self.complete_transactions(allocations)
        self.bids.clear()
        return allocations # TODO returning for testing. Do we need this? Does it interfere with main code?

    def complete_transactions(self, allocations):
        for allocation in allocations:
            allocation.sale_property.realized_price = allocation.final_price
            # print(f'Time {self.model.time_step}, Property {allocation.property.unique_id}, Price {allocation.property.realized_price}')
            if isinstance(allocation.buyer, Investor):
                self.handle_investor_purchase(allocation)
            elif isinstance(allocation.buyer, Person):
                self.handle_person_purchase(allocation)
            else:
                logger.warning('Buyer was neither a person nor an investor.')
            if isinstance(allocation.seller, Person):
                self.handle_seller_departure(allocation)
            else:
                logger.warning('Seller was not a person, so was not removed from the model.')

    def handle_investor_purchase(self, allocation):
        """Handles the purchase of a property by an investor."""
        self.transfer_ownership(allocation.buyer, allocation.seller, allocation.sale_property)
        logger.debug('Property %s sold to investor.', allocation.sale_property.unique_id)
        self.rental_listing.append(allocation.sale_property)

    def handle_person_purchase(self, allocation):
        """Handles the purchase of a property by a person."""
        self.transfer_ownership(allocation.buyer, allocation.seller, allocation.sale_property)
        allocation.sale_property.resident = allocation.buyer
        allocation.buyer.residence = allocation.sale_property
        self.model.grid.move_agent(allocation.buyer, allocation.sale_property.pos)
        logger.debug('Property %s sold to newcomer %s.', allocation.sale_property.unique_id, allocation.buyer.unique_id)

        if allocation.buyer.unique_id in self.workforce.newcomers:
            self.workforce.remove(allocation.buyer, self.workforce.newcomers)
        # print(f'Time {self.model.time_step} New worker {buyer.unique_id} Loc {sale_property}') # TEMP

    def handle_seller_departure(self, allocation):
        """Handles the departure of a selling agent."""
        if allocation.seller.unique_id in self.workforce.retiring:
            # print('seller removed')
            allocation.seller.remove()
        else:
            logger.warning('Seller was not retiring, so was not removed from the model.')

    def transfer_ownership(self, buyer, seller, sale_property):
        """Transfers ownership of the property from seller to buyer."""
        buyer.properties_owned.append(sale_property)
        seller.properties_owned.remove(sale_property)
        sale_property.owner = buyer

    def rent_homes(self):
        """Rent homes listed by investors to newcomers."""
        logger.debug(f'{len(self.rental_listing)} properties to rent.')
        # print(len(self.rental_listing))
        for rental in self.rental_listing:
            renter = self.model.create_newcomer()
            rental.resident = renter
            renter.residence = rental
            self.workforce.remove(renter, self.workforce.newcomers)
            logger.debug(f'Newly created renter {renter.unique_id} lives at '
                         f'property {renter.residence.unique_id} which has '
                         f'resident {rental.resident.unique_id}.')
        self.rental_listing.clear()

class Bid:
    def __init__(
        self, 
        bidder: Union[Person, Investor],
        sale_property: Land, 
        price: Union[float, int],
        bid_type: str = "",
        mortgage: Union[float, int] = 0.0,
    ):
        if not isinstance(bidder, (Person, Investor)):
            raise ValueError("Bidder must be of type Person.")  
        if not isinstance(sale_property, Land):
            raise ValueError("Property must be of type Land.")
        if not isinstance(price, (float, int)):
            raise ValueError("Price must be a numeric value.")
        if not isinstance(bid_type, (str)):
            raise ValueError("Price must be a numeric value.")
        if not isinstance(mortgage, (float, int)):
            raise ValueError("Mortgage must be a numeric value.")
               
        self.bidder = bidder
        self.sale_property = sale_property
        self.price = price
        self.mortgage = mortgage
        self.bid_type = type

    def __str__(self):
        return f"Bidder: {self.bidder}, Property: {self.sale_property}, Price: {self.price}, Mortgage: {self.mortgage}, Type: {self.type}"

class Allocation:
    def __init__(
        self, 
        buyer: Union[Person, Investor],
        seller: Union[Person, Investor],
        sale_property: Land,
        final_price: Union[float, int] = 0.0,
        highest_bid: Union[float, int] = 0.0,
        second_highest_bid: Union[float, int] = 0.0,
    ):
        if not isinstance(buyer, (Person, Investor)):
            raise ValueError("Successful bidder must be of type Person or Investor.")
        if not isinstance(seller, (Person, Investor)):
            raise ValueError("Successful bidder must be of type Person or Investor.")
        if not isinstance(sale_property, Land):
            raise ValueError("Property must be of type Land.")
        if not isinstance(final_price, (float, int)):
            raise ValueError("Final price must be a numeric value.")
        if not isinstance(highest_bid, (float, int)):
            raise ValueError("Highest bid must be a numeric value.")
        if not isinstance(second_highest_bid, (float, int)):
            raise ValueError("Second highest bid must be a numeric value.")

        self.buyer              = buyer
        self.seller             = seller
        self.sale_property      = sale_property
        self.final_price        = final_price
        self.highest_bid        = highest_bid
        self.second_highest_bid = second_highest_bid

    def __str__(self):
        return f"Property: {self.sale_property}, Successful Bidder: {self.buyer}, Highest Bid: {self.highest_bid}, Second Highest Bid: {self.second_highest_bid}, Final Price: {self.final_price}"
