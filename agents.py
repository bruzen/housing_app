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

    # TODO do we update warranted rent and prices in the LAND STEP?
    @property
    def warranted_rent(self):
        omega  = self.model.firm.wage_premium
        psi    = self.model.subsistence_wage
        a      = self.model.housing_services_share
        cd     = self.transport_cost
        # c      = self.model.transport_cost_per_dist
        # d      = self.distance_from_center
        return omega - cd + a*psi

    @property 
    def market_rent(self):
        # TODO this can change
        return self.warranted_rent

    @property
    def net_rent(self):
        """Compute the rent for a land parcel, or what someone could afford
        to pay to live there. 

        Rent depends on the urban wage premium over and above the subsistence
        wage, and on transportation costs and the distance to the
        central business district. Applies with a single wage. Adjust for
        differential urban wages.

        omega - c*d + a*psi - b*a*psi - tau*a*psi

        """
        return self.warranted_rent - self.maintenance - self.property_tax

    @property
    def warranted_price(self):
        return self.warranted_rent/self.model.r_prime
    
    @property
    def appraised_price(self):
        # Property valued used for taxation purposes
        # TODO this should be a lagged market price
        return self.warranted_price

    @property
    def property_tax(self):
        tau              = self.property_tax_rate
        appraised_price  = self.appraised_price
        return tau * appraised_price
    
    # def get_tax(self):
    #     """ 
    #     THIS DOES NOT CHANGE WITH INCREASING WAGES?
    #     BUT THAT IS THE MAIN WAY TO FUND A CITY

    #     WHAT TO CALL THIS WEHRE DOES IT GO. WHERE DO WE USE THIS VS TAU
    #     Just for initialization? - warranted price. 
    #     Use warranted prices as initialization
    #     Tax costs for the mortgage period, T. 
    #     (Example of rate for an  multiperiod annual rate)
    #     tax_T= tau*(omega-c*d + a*psi) * sum_delta_T
    #     This is assuming taxes are paid at the end of each year for T years
    #     tau_T       = tau * sum_T_delta 
    #     #  present value of the tax rate over T years        
    #     """
    #     tau   = self.property_tax_rate
    #     omega = self.model.firm.wage_premium
    #     psi   = self.model.subsistence_wage
    #     a     = self.model.housing_services_share
    #     c     = self.model.transport_cost_per_dist # RENAME
    #     d     = self.distance_from_center
    #     sum_delta = self.model.discount_factor
    #     return tau * (omega - c*d + a*psi) * sum_delta


    @property
    def maintenance(self):
        a      = self.model.housing_services_share
        b      = self.model.maintenance_share
        psi    = self.model.subsistence_wage
        return a * b * psi
    
    # TODO consider sum_delta
    #     """Maintenance share of property service (a*b*psi summed and discounted)
    #     OR IS IT TOTAL maintenance COST OVER THE MORTGAGE PERIOD?
    #     """
    #     sum_delta = self.discount_factor # Could vary by person
    #     return (a * b * psi) * sum_delta

    def __init__(self, unique_id, model, pos, 
                 property_tax_rate = 0., 
                 resident = None, owner = None):
        super().__init__(unique_id, model)
        self.pos                  = pos
        self.property_tax_rate    = property_tax_rate
        self.resident             = resident
        self.owner                = owner

        # TODO want warranted price history? - 
        # self.rent_history         = [] 
        self.offers               = []
        self.euclidean_dist_from_center = distance.euclidean(self.pos,
                                                             self.model.center)
        self.cityblock_dist_from_center = distance.cityblock(self.pos,
                                                             self.model.center)
        self.distance_from_center =  self.euclidean_dist_from_center
        # `transport_cost` cacluates travel costs using euclidean distance.
                
        self.transport_cost       = (self.distance_from_center *
                                     self.model.transport_cost_per_dist)

    def step(self): 
        # self.rent_history.append(self.net_rent)
        # self.tax = self.get_tax()
        # How to handle realized prices?
        price_data = {'id': self.unique_id, 
                           'warranted_price': self.warranted_price,
                           'time_step': self.model.time_step,
                           'transport_cost': self.transport_cost,
                           'wage': self.model.firm.wage}
        self.model.step_price_data.append(price_data)

class Person(Agent):
    """Person.

    :param unique_id: An integer identifier.
    :param model: The main city model.
    :param pos: The person's location on the spatial grid.
    :param init_working_period: The initial working period, between 0
        and the retirement age. It increments in each step.
    :param savings: The amount of money the agent has in savings.
    :param debt: The amount of money the agent owes.
    :param residence: The land parcel where the agent lives.
    """

    @property
    def borrowing_rate(self):
        return self.model.r_target + self.individual_wealth_adjustment

    @property
    def individual_wealth_adjustment(self):
        return 5

    def __init__(self, unique_id, model, pos, init_working_period = 0,
                 savings = 0., debt = 0.,
                 residence_owned = None):
        super().__init__(unique_id, model)
        self.pos = pos

        self.init_working_period = init_working_period
        self.working_period      = init_working_period
        self.savings             = savings
        # self.debt                = debt # TODO make mortgage

        self.properties_owned    = []
        self.residence           = residence_owned

        self.bank                = self.model.bank 

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

    # @property
    # def wealth(self):
    #     # TODO fix wealth is a function of assets and income
    #     return -1
    
    # @property
    # def r(self):
    #     # TODO fix r should be a function of wealth
    #     return self.bank.r_prime

    def step(self):
        self.count              += 1
        self.working_period     += 1

        # Update savings
        self.savings += self.model.savings_per_step

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
                premium = self.model.firm.wage - self.model.subsistence_wage
                if (premium > self.residence.transport_cost):
                    self.is_working = 1

            # TODO pay costs for any properties owned
            # if self.residence in self.properties_owned:
            #     # TODO pay mortgage if needed pay costs
            #     pass
            # else:
            #     self.savings -= self.rent # TODO check this is right rent

        elif self in self.model.retiring_agents:
            logger.debug(f'Retiring agent {self.unique_id} still in model.')

        else:
            logger.debug(f'Agent {self.unique_id} has no residence.')
    
    def bid(self):
        """Newcomers bid on properties for use or investment value."""
        # TODO add a percentage downpayment
        # TODO the lower bound on the bid price is approved financing
        # TODO make a parameter for downpayment requirement, 
        # eg 20% for all initially
        # TODO: FIX/THINK ABOUT bid. Some people will pay 
        # down more than the min from their savings.
        # TODO confirm they have enough savings
        max_allowed_bid = self.bank.get_max_allowed_bid(self)
        for sale_property in (self.model.realtor.sale_listing):
            max_desired_bid = self.model.bank.get_max_desired_bid(sale_property, self)
            max_bid = min(max_allowed_bid, max_desired_bid)
            offer = {'Bidder': self,
                     'Offer':  max_bid}
            offer_val = offer['Offer']
            logger.debug(f'Person {self.unique_id} bids {offer_val} \
                           for property {sale_property.unique_id}, \
                           if val is positive.')
            if offer_val > 0:
                sale_property.offers.append(offer)

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

    # # TODO maybe move Firm above Person
    # @property
    # def no_workers(self):
    #     # TODO check we always add/remove from working list properly
    #     return len(self.workers)
    
    @property
    def workforce_urban_firm(self):
    #   TODO FIX THIS NEEDS TO UPDATE
        return self.model.workforce_rural_firm

    @property
    def no_firms(self):
    #     TODO FIX THIS NEEDS TO UPDATE
    #     TODO- initialize with baseline pop and rural workers/firm
    #     return population/workers_per_firm
        return self.model.baseline_population/self.model.workforce_rural_firm

    def __init__(self, unique_id, model, pos, init_wage_premium,
                 alpha_firm, beta_firm, price_of_output, cost_of_capital,
                 wage_adjust_coeff_new_workers, 
                 wage_adjust_coeff_exist_workers):
        super().__init__(unique_id, model)
        self.pos             = pos
        self.wage_premium    = init_wage_premium # omega
        self.wage            = init_wage_premium + self.model.subsistence_wage
        self.alpha_firm      = alpha_firm
        self.beta_firm       = beta_firm
        self.price_of_output = price_of_output
        self.cost_of_capital = cost_of_capital
        self.wage_adjust_coeff_new_workers   = wage_adjust_coeff_new_workers
        self.wage_adjust_coeff_exist_workers = wage_adjust_coeff_exist_workers

    # TODO Fix Firm wage update totaly and move to model
    def step(self):
        # prefactor  = self.model.prefactor
        # agglom     = self.model.agglomeration_ratio
        # population = self.model.agglomeration_population
        # workers_share = self.model.workers_share  # lambda - TODO fix
        # wage_premium = workers_share * (agglom-1) * prefactor * population**agglom # omega # ****** 
        # self.wage = wage_premium + self.model.psi

        # k thought # self.wage_premium = (workers_share * prefactor * population**agglom)/ population # omega    
        # note surplus is: (beta - 1) * (prefactor * population**agglom)        
        self.wage_premium += 0.01
        self.wage += 1 
        # self.wage_premium   = wage_premium # **** TODO UPDATE URBAN WAGE PREMIUM
        # logger.error(f'Wage {self.wage}') # TODO Temp
        

class Bank(Agent):
    """Bank.

    :param unique_id: An integer identifier.
    :param model: The main city model.
    :param pos: The bank's location on the spatial grid.
    :param r_prime interest_rate: The prime interest rate offered by the bank.
    # OLD :param borrowing_ratio: The borrowing ratio permitted by the bank.
    # TODO change
    :param properties_owned: Properties owned by the bank initially. 
    # TODO owned by bank will not be initialized to 0, 
    # TODO do what we did with the bank
    #:param property_management_costs: TODO fix/replace with several terms for 
    taxes/maintenance/etc
    # :param savings: TODO fix/replace this may not be necessary
    # :param debt: TODO fix/replace
    # :param loans: Loans owned to the bank 
    # TODO loans may be objects with a borrower, amount, rate, term etc
    # :param rent_ratio: rent per period as a share of the rent paid, always 1
    # :param opperations_ratio: opperating costs as a share of rent paid
    # :param tax_ratio: taxes as a share of rent paid
    """

    def __init__(self, unique_id, model, pos,
                 r_prime = 0.05, max_mortgage_share = 0.9,
                 properties_owned = [],
                 # savings = 0., # debt = 0., loans = 0.,
                 ):
        super().__init__(unique_id, model)
        self.pos = pos

        # property_management_costs = -1.
        # Properties for bank as a lender
        self.r_prime             = r_prime
        self.max_mortgage_share  = max_mortgage_share
        self.min_downpayment_share = 0.2

        # Properties for bank as an asset holder
        # self.property_management_costs = property_management_costs # TODO 
        self.properties_owned    = properties_owned
        # self.savings             = savings # TODO do banks have savings?
        # self.debt                = debt
        # self.loans               = loans

    def bid(self):
        """Banks bid on investment properties."""
        for sale_property in (self.model.realtor.sale_listing):
            # FIX - THIS NO LONGER INCLUDES BANK MAINTENANCE FACTORS ETC. - 
            max_desired_bid = self.model.bank.get_max_desired_bid(sale_property, self)
            offer = {'Bidder': self,
                     'Offer':  max_desired_bid}
            offer_val = offer['Offer']
            logger.debug(f'Bank {self.unique_id} bids {offer_val} for \
                         property {sale_property.unique_id}, if val is positive.')
            if offer_val > 0:
                sale_property.offers.append(offer)

    def get_max_allowed_bid(self, applicant):
        savings = applicant.savings 
        return savings # TODO placeholder - fix 
        # REPLACES GET MAX MORTGAGE
        # FIX min_downpayment = self.bank.min_down_payment_share * max_mortgage
        # self.bank.max_mortgage_share
        # downpayment = min(min_downpayment, self.savings)
        # max_allowed_bid = max_mortgage + downpayment
        # if applicant in self.model.schedule.get_breed_agents(Person):
        #     wage = self.model.firm.wage
        #     i    = self.get_mortgage_interest_rate(applicant)
        #     if i > 0:
        #         # TODO mortgage should be based on savings not just wage.
        #         max_mortgage = self.borrowing_ratio * wage / i 
        #     else:
        #         logger.warning(f'Max_mortgage calculation requires greater  \
        #                        than zero interest rate, but interest is {i} \
        #                        for agent {applicant.unique_id}.')
        #         max_mortgage = None
        # else:
        #     logger.warning(f'Max_mortgage calculation applies for a person. \
        #                    Applicant {applicant.unique_id} is not one.')
        #     max_mortgage = None 

    def get_max_desired_bid(self, property, bidder): # ADD downpayment , downpayment):
        """Compute the perceived investment value of a property for
        a particular agent.

        The investor can charge rent and capture a growing stream of rents
        as the city grows. They may borrow money at a given interest rate and
        incur maintenance costs, fees, and taxes. 

        The value of a property depends on the individual's individual cost 
        benefit analysis as well as on perceived risks and individual's
        risk aversion.

        Investors who will live in a property also benefit from it's 
        use value, which may be compared against the cost of renting.

        R / (r - p_dot)
        R rent today
        r discount rate, roughly equivalent to the interest rate for banks
        p_dot rate of rental price growth

        :param property: the land parcel to evaluate.
        :param investor: the agent considering purchasing in a property.

        """
        net_rent = property.net_rent
        r        = self.model.r_prime # self.get_mortgage_interest_rate(investor)
        r_target = self.model.r_target
        m        = self.model.max_mortgage_share # if it is a bank # 0.8 # TODO FIX - ADD WEALTH mortgage share. depends on price.
        # m is the downpayment they are proposing to make (downpayment share?)
        delta    = self.model.discount_factor 
        p_dot    = self.model.p_dot # Rate of price change
        return net_rent/((1 - m)*r_target - delta*(1 + p_dot - (1 + r)*m))
    
        # TODO the investor must be a person or a bank initially
        # TODO Consider alternative discount rates for individuals    
        # TODO make sure demoninator for the value is not zero
        # OLD CALCULATIONS
        # value = rent / (r - p_dot)
        # net_revenue    = self.get_net_revenue()
        # rA             = self.model.r_target
        # rM             = self.get_mortgage_interest_rate(buyer)
        # return forecast_price * (p_dot - rA + net_revenue) / (1 + rM*m)


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

    def add_bid(self, bidder_id, property_id, price):
        if bidder_id not in self.bidders:
            self.bidders[bidder_id] = Bidder(bidder_id)

        if property_id not in self.properties:
            self.properties[property_id] = Property(property_id)

        bidder = self.bidders[bidder_id]
        property = self.properties[property_id]

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

            if isinstance(buyer, Bank):
                self.handle_bank_purchase(buyer, allocation.property)
            elif isinstance(buyer, Person):
                self.handle_person_purchase(buyer, allocation.property, final_price)
            else:
                logger.warning('Buyer was neither a person nor a bank.')

            if isinstance(seller, Person):
                self.handle_seller_departure(seller)
            else:
                logger.warning('Seller was not a person, so was not removed from the model.')

    def transfer_property(self, seller, buyer, sale_property):
        """Transfers ownership of the property from seller to buyer."""
        buyer.properties_owned.append(sale_property)
        seller.properties_owned.remove(sale_property)
        sale_property.owner = buyer

    def handle_bank_purchase(self, buyer, sale_property):
        """Handles the purchase of a property by a bank."""
        logger.debug('Property %s sold to bank.', sale_property.unique_id)
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
        """Rent homes listed by banks to newcomers."""
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
    def __init__(self, bidder, property, price):
        self.bidder = bidder
        self.property = property
        self.price = price

class Allocation:
    def __init__(self, property, successful_bidder, original_price, final_price):
        self.property = property
        self.successful_bidder = successful_bidder
        self.original_price = original_price
        self.final_price = final_price
