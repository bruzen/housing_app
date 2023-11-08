# TODO List

## High Priority 
- [***] Why is is_working crazy in heatmaps? Why does city not grow? With no owner investors (all People buyers), some people in the city (maybe after sale) do not work. Do people move in and start working after buying - do any of them get removed by accident. (Maybe handle_person_purchase - should make sure it results in workers - not all do, but some? maybe. why?).  Realized population looks wrong?
- [ ] Show most recent realized_price in heatmap, with x on top if just sold
- [ ] Why are newcomer agents bidding 0? Maybe they need savings.
- [ ] Add distributions for newcomer initial savings and mean wealth calculations for wealth adjustment
- [ ] Fix p-dot
- [ ] Bidding
- [ ] Consider why bids are so far off warranted price
- [ ] Add add worthwhile to work check to warranted price


## Medium Priority

- [ ] Track savings
- [ ] Implement mortgages


## Low Priority

- [ ] Add error checking to fail gracefully if no p_dot or no realized price data.
- [ ] Use one plotting library to speed imports
- [ ] If a property is listed for sale, calculate sale properties - calc warranted_rent etc. to speed up (either for all properties or when listed)
- [ ] Record Workforce objects count of workers, newcomers, retiring as a sanity check
- [ ] Check init_F
- [ ] Calc/record rent_captured_by finance, share_captured_by_finance, and urban_surplus
- [ ] Make a hover over that shows person data for the heatmap - do I store more in person - id/x/y
- [ ] Fix test_auction for new plot? iPython notebook used some error messages I removed
- [ ] Display bid type (e.g. equity_limited etc)
- [ ] Consider sigmoid transitions for get_bid transition
  

## DiR

- [ ] Implement individual_wealth_adjustment
- [ ] In bid calc realized_price: make price model use realized_price, around line 674
- [ ] Include warranted price, and effect of more bids in shaping final price
- [ ] How many newcomers to create. Maybe draw 50 items from the distribution and take the max from those drawn?  - TODO - number of bids is meaningless--. as are low bids because I'm just making incoming agents based on the number of properties for sale. (Could be a distribution of potential newcomers instead of full agents - just some data structure with the initial savings etc.)
- [ ] WHY ARE BIDS 0
- [ ] Talk through:
        self.rent_captured_by_finance  = 0 # TODO implement. make a marker for agents in the city
        self.share_captured_by_finance = 0 # TODO implement.
        self.urban_surplus   = 0 # TODO implement
- [ ] 

## Orienting

- [ ] Search TODO
- [ ] Review TO SORT tasks below
- [ ] Trace flow - think about possible errors


## Consider for Future Work

- [ ] Do we only count amenity for workers, or those in the urban boundary?
- [ ] Add capital gains, tax first time buyer subsidy -  Add transaction costs in WTP - it is the unfair floor that is the piece tht does the work.
- [ ] Add differential income
- [ ] How to do the resilience experiment?
- [ ] Size without population and/or density in different areas. 
- [ ] Calcule mean, median, or the percentage change over time, regression for direction and strength of the trends.


## Check

- [x] Fix - WARNING: Seller was not a person, so was not removed from the model.
- [x] Land step - # TODO if residence is not owned, renter moves out
- [x] Fix retiring renter exit: 
      if (self.residence in self.properties_owned):
	 TODO Contact bank. Decide: sell, rent or keep empty
        self.model.realtor.sale_listing.append(self.residence)
	 TODO if residence is not owned, renter moves out
- [x] Add renters after sale
- [x] Manage retirement at the edge of the city 
- [x] Fix logging
- [x] Add renters after sale -- show they are renters vs owners in heatmap
- [x] 239 TODO if residence is not owned, renter moves out
- [x] Fix get_max_bid
- [x] Weird now warranted price is flat and realized not - or on top?
- [ ] FIX TO IN WORKERS-- RECORD WORKERS, NEWCOMERS, RETIRING

## Done

- [x] Make investors bid on properties
- [x] DONE do we want to name the pieces of this:  m = max_mortgage_share - 0.1 * average_wealth / W  Name 0.1 lenders_wealth_sensitivity