import math
import logging
from collections import defaultdict
from scipy.spatial import distance
import numpy as np
import pandas as pd

from mesa import Agent

logging.basicConfig(filename='logfile.log',
                    filemode='w',
                    level=logging.ERROR,
                    format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

class Land(Agent):
    """Land parcel.

    :param unique_id: An integer identifier.
    :param model: The main city model.
    :param pos: The land parcel's location on the spatial grid.
    :param resident: The agent who resides at this land parcel.
    :param owner: The agent who owns this land parcel.
    # :param rent_history: The history of rent values for the land parcel.
    """

    @property
    def warranted_rent(self):
        omega     = self.model.firm.wage_premium
        psi       = self.model.subsistence_wage
        a         = self.model.housing_services_share
        cd        = self.transport_cost
        return omega - cd + a * psi

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
        a         = self.model.housing_services_share
        b         = self.model.maintenance_share
        psi       = self.model.subsistence_wage
        return a * b * psi

    def __init__(self, unique_id, model, pos, 
                 property_tax_rate = 0., 
                 resident = None, owner = None):
        super().__init__(unique_id, model)
        self.pos                  = pos
        self.property_tax_rate    = property_tax_rate
        self.resident             = resident
        self.owner                = owner

        self.offers               = []
        self.distance_from_center = self.calculate_distance_from_center()
        self.transport_cost       = self.calculate_transport_cost()

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
 
        Formula: r_target + K/(W-W_min) - K/(W_mean-W_min)
        K is wealth sensitivity parameter

        Returns:
        The individual wealth adjustment value.
        """
        # r_target = self.model.r_target
        # K        = self.model.wealth_sensitivity
        # W        = self.get_wealth() 
        # W_min
        # W_mean
        return 0.0002

    def __init__(self, unique_id, model, pos, init_working_period = 0,
                 savings = 0., debt = 0.,
                 residence_owned = None):
        super().__init__(unique_id, model)
        self.pos = pos

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
        self.is_working          = 0

    def step(self):
        self.count              += 1
        self.working_period     += 1


        # Newcomers, who don't find a home, leave the city
        if (self in self.model.newcomers):
            if (self.residence == None):
                if (self.count > 0):
                    logger.debug(f'Newcomer {self.unique_id} removed, who \
                                   owns {self.properties_owned}.')
                    self.remove_agent()
            else:
                logger.error(f'Newcomer {self.unique_id} has a \
                               residence {self.residence.unique_id}, \
                               but was not removed from newcomer list.')

        elif (self.residence) and (self not in self.model.retiring_agents):
            # Retire if past retirement age
            if (self.working_period > self.model.working_periods):
                self.model.retiring_agents.append(self)
                # List homes for sale
                if (self.residence in self.properties_owned):
                    # TODO Contact bank. Decide: sell, rent or keep empty
                    self.model.realtor.sale_listing.append(self.residence)
                    # TODO if residence is not owned, renter moves out

            # Work if it is worthwhile to work
            else:
                self.is_working     = 0
                # TODO: check same calc as city_extent. Remove redundancy.
                premium = self.model.firm.wage_premium
                if (premium > self.residence.transport_cost):
                    self.is_working = 1

            # Update savings
            self.savings += self.model.savings_per_step # TODO debt, wealth
            self.wealth  = self.get_wealth()

        elif self in self.model.retiring_agents:
            logger.debug(f'Retiring agent {self.unique_id} still in model.')

        else:
            logger.debug(f'Agent {self.unique_id} has no residence.')

    def bid(self):
        """Newcomers bid on properties for use or investment value."""
        
        W = self.get_wealth() # TODO use wealth in mortgage share and borrowing rate
        S = self.savings
        r = self.borrowing_rate
        m = self.get_max_mortgage_share()
        M = self.get_max_mortgage()
        
        r_target = self.model.r_target # TODO this is personal but uses same as bank. Clarify.
        

        for sale_property in self.model.realtor.sale_listing:

            R_N = sale_property.net_rent
            P_max_bid = self.model.bank.get_max_bid(R_N, r, r_target, m)

            if m * P_max_bid < m:
                mortgage = m * P_max_bid
                P_bid = min(m * P_max_bid + S, P_max_bid)

            else:
                mortgage = M
                P_bid = min(M + S, P_max_bid)

            bid = Bid(bidder=self, property=sale_property, price=P_bid, mortgage=mortgage)
            logger.debug(f'Person {self.unique_id} bids {bid.price} \
                        for property {sale_property.unique_id}, \
                        if val is positive.')
            if bid.price > 0:
                sale_property.offers.append(bid)

    def get_wealth(self):
        # TODO Wealth is properties owned, minuse mortgages owed, plus savings.
        return self.savings

    def get_max_mortgage_share(self):
        return 0.8
    
    def get_max_mortgage(self):
        S        = self.savings
        r        = self.borrowing_rate
        r_prime  = self.model.r_prime
        wage     = self.model.firm.wage
        return 0.28 * (wage + r * S) / r_prime

    def remove_agent(self):
        if self in self.model.newcomers:
            self.model.newcomers.remove(self)
        if self in self.model.retiring_agents:
            self.model.retiring_agents.remove(self)
        self.model.grid.remove_agent(self)
        self.model.schedule.remove(self)

class Firm(Agent):
    """Firm.

    :param unique_id: An integer identifier.
    :param model: The main city model.
    :param pos: The firms's location on the spatial grid.
    :param init_wage_premium: initial urban wage premium.
    """

    # TODO do both include seed_population?
    @property
    def N(self):
        """total_no_workers"""
        total_no_workers = len(
            [a for a in self.model.schedule.agents_by_breed[Person].values()
                     if a.is_working == 1]
        )
        return total_no_workers * self.model.density + self.model.seed_population

    @property
    def wage(self):
        return self.wage_premium + self.model.subsistence_wage

    def __init__(self, unique_id, model, pos, init_wage_premium,
                 alpha_F, beta_F, Z,
                 price_of_output, cost_of_capital,
                 firm_adjustment_parameter,
                 wage_adjustment_parameter):
        super().__init__(unique_id, model)
        self.pos             = pos
        self.wage_premium    = init_wage_premium # omega
        self.alpha_F         = alpha_F
        self.beta_F          = beta_F
        self.Z               = Z

        self.price_of_output = price_of_output
        self.r               = cost_of_capital

        self.firm_adjustment_parameter = firm_adjustment_parameter
        self.wage_adjustment_parameter = wage_adjustment_parameter

        n_R           = self.model.workforce_rural_firm
        self.n        = n_R # workforce_urban_firm is initally same as urban firm

        self.F        = 1. # TODO INITIALIZE, CHECK IN OVERLEAF
        # self.no_firms = self.model.baseline_population/self.model.workforce_rural_firm

        # Calculate scale factor A for a typical urban firm
        psi      = self.model.subsistence_wage
        Y_R      = n_R * psi / beta_F
        Y_U      = self.n * self.wage / beta_F
        k_R      = alpha_F * Y_R / self.r
        self.k   = alpha_F * Y_U / self.r
        self.A_F = Y_R/(k_R**alpha_F * n_R * psi**beta_F)

    def step(self):
        # Calculate wage, capital, and firm count given number of urban workers
        self.n = self.N/self.F
        y = self.output(self.N, self.k, self.n)

        MPL = self.beta_F  * y / self.n
        MPK = self.alpha_F * y / self.k

        n_target = self.beta_F * y / self.wage
        y_target = self.output(self.N, self.k, n_target)
        k_target = self.alpha_F * y_target / self.r

        # N_target_exist = n_target/self.n * self.N
        adj_f = self.firm_adjustment_parameter
        F_target = n_target/self.n * self.F
        F_next = (1 - adj_f) * self.F + adj_f * F_target
        N_target_total = F_next * n_target
        F_next_total = N_target_total / n_target

        # adj_l = 1.25 # TODO self.labor_adjustment_parameter
        # N_target_total = adj_l * n_target/self.n * self.N
        # N_target_new = n_target * self.Z * (MPL - self.wage)/self.wage * self.F # TODO - CHECK IS THIS F-NEXT?

        c = self.model.transport_cost_per_dist
        wage_premium_target = c * math.sqrt(N_target_total/(2*self.model.density))        
        

        k_next = k_target # TODO fix

        adj_w = self.wage_adjustment_parameter
        self.wage_premium = (1-adj_w)*self.wage_premium + adj_w * wage_premium_target
        self.k = k_next
        self.F = F_next_total # OR use F_total

    def output(self, N, k, n):
        A_F     = self.A_F
        alpha_F = self.alpha_F
        beta_F  = self.beta_F
        gamma   = self.model.gamma

        return A_F * N**gamma * k**alpha_F * n**beta_F

class Bank(Agent):
    def __init__(self, unique_id, model, pos,
                 r_prime = 0.05, max_mortgage_share = 0.9,
                 ):
        super().__init__(unique_id, model)
        self.pos = pos

    def get_max_bid(self, R_N, r, r_target, m):
        T      = self.model.mortgage_period
        delta  = self.model.delta
        p_dot  = self.model.get_p_dot()

        # if R_N is None:
        #     print("Value R_N is None.")
        # if r is None:
        #     print("Value r is None.")
        # if r_target is None:
        #     print("Value r_target is None.")
        # if m is None:
        #     print("Value m is None.")
        # if p_dot is None:
        #     print("Value p_dot is None.")

        # if R_N is not None and r is not None and r_target is not None and m is not None and p_dot is not None:
        R_NT   = ((1 + r)**T - 1) / r * R_N
        return R_NT / ((1 - m) * r_target/(delta**T) - p_dot)

class Investor(Agent):

    @property
    def borrowing_rate(self):
        self.model.r_target
    
    def __init__(self, unique_id, model, pos, properties_owned = []):
        super().__init__(unique_id, model)
        self.pos = pos

        # Properties for bank as an asset holder
        # self.property_management_costs = property_management_costs # TODO 
        self.properties_owned      = properties_owned

    # def step(self):
    #     self.bid()

    def bid(self):
        # """Investors bid on investment properties."""
        pass
        # m = 0.9 # TODO fix
        # r = self.borrowing_rate
        # r_target = self.model.r_target
        
        # for sale_property in self.model.realtor.sale_listing:
        #     R_N = sale_property.net_rent
        #     print(R_N)
        #     P_max_bid = self.model.bank.get_max_bid(R_N, r, r_target, m)
        #     mortgage = m * P_max_bid
        #     bid = Bid(bidder=self, property=sale_property, price=P_max_bid, mortgage=mortgage)
        #     logger.debug(f'Bank {self.unique_id} bids {bid.price} for \
        #                 property {sale_property.unique_id}, if val is positive.')
        #     if bid.price > 0:
        #         sale_property.offers.append(bid)
    
class Realtor(Agent):
    """Realtor agents connect sellers, buyers, and renters."""
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos

        self.sale_listing = []
        self.rental_listing = []

        self.bidders = {}
        self.properties = {}
        self.bids = defaultdict(list)

    def step(self):
        pass

    def add_bid(self, bidder, property, price):
        if bidder not in self.bidders:
            self.bidders[bidder] = bidder

        if property not in self.properties:
            self.properties[property] = property
    
        bid = Bid(bidder, property, price)
        self.bids[property].append(bid)

    def sell_homes(self):
        allocations = []

        for property in self.bids.keys():
            property_bids = self.bids[property]
            property_bids.sort(key=lambda x: x.price, reverse=True)

            highest_bid = property_bids[0]
            second_highest_price = property_bids[1].price if len(property_bids) > 1 else 0

            if highest_bid.price > second_highest_price:
                allocation = Allocation(property, highest_bid.bidder, highest_bid.price, second_highest_price)
            else:
                allocation = Allocation(property, None, highest_bid.price, 0)

            allocations.append(allocation)

        self.complete_transactions(allocations)

        self.bids.clear()
        self.bidders.clear()
        self.properties.clear()

    def complete_transactions(self, allocations):
        for allocation in allocations:
            buyer = allocation.successful_bidder
            seller = allocation.property.owner
            final_price = allocation.final_price

            self.transfer_property(seller, buyer, allocation.property)

            if isinstance(buyer, Investor):
                self.handle_investor_purchase(buyer, allocation.property)
            elif isinstance(buyer, Person):
                self.handle_person_purchase(buyer, allocation.property, final_price)
            else:
                logger.warning('Buyer was neither a person nor an investor.')

            if isinstance(seller, Person):
                self.handle_seller_departure(seller)
            else:
                logger.warning('Seller was not a person, so was not removed from the model.')

    def transfer_property(self, seller, buyer, sale_property):
        """Transfers ownership of the property from seller to buyer."""
        buyer.properties_owned.append(sale_property)
        seller.properties_owned.remove(sale_property)
        sale_property.owner = buyer

    def handle_investor_purchase(self, buyer, sale_property):
        """Handles the purchase of a property by an investor."""
        logger.debug('Property %s sold to investor.', sale_property.unique_id)
        self.rental_listing.append(sale_property)

    def handle_person_purchase(self, buyer, sale_property, final_price):
        """Handles the purchase of a property by a person."""
        sale_property.resident = buyer
        buyer.residence = sale_property
        self.model.grid.move_agent(buyer, sale_property.pos)
        logger.debug('Property %s sold to newcomer %s.', sale_property.unique_id, buyer.unique_id)

        if buyer in self.model.newcomers:
            self.model.newcomers.remove(buyer)

    def handle_seller_departure(self, seller):
        """Handles the departure of a selling agent."""
        if seller in self.model.retiring_agents:
            seller.remove_agent()
        else:
            logger.warning('Seller was not retiring, so was not removed from the model.')

    def rent_homes(self):
        """Rent homes listed by investors    to newcomers."""
        logger.debug(f'{len(self.rental_listing)} properties to rent.')
        for rental in self.rental_listing:
            renter = self.model.create_newcomer()
            rental.resident = renter
            renter.residence = rental
            self.model.newcomers.remove(renter)
            logger.debug(f'Newly created renter {renter.unique_id} lives at '
                         f'property {renter.residence.unique_id} which has '
                         f'resident {rental.resident.unique_id}.')
        self.rental_listing.clear()

class Bid:
    def __init__(self, bidder, property, price, mortgage=0.):
        self.bidder   = bidder
        self.property = property
        self.price    = price
        self.mortgage = mortgage

class Allocation:
    def __init__(self, property, successful_bidder, bid_price, final_price):
        self.property          = property
        self.successful_bidder = successful_bidder
        self.bid_price         = bid_price
        self.final_price       = final_price