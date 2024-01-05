
# TODO List

## High Priority
- [ ] Look for cache/memory problems. Try pylint
- [ ] Make it run on the command line. See if that works on AWS
- [ ] Make a branch to record differences (Check it runs as is, commit, stash, branch)
- [ ] Make a branch with changes for  track-bid-details
- [ ] Generate evenly spaced grid to iterate over
- [ ] Param sweep for capital gains tax people/institutions .1 to .4  (or 0-1). Plot final ownership share, and ownership share every 10 steps - they fall at a different point .. what is the point of fall?
 - [ ] boundary search - contour plots - capital gains tax - capital gains tax institutions.. 
- [ ] * perturb mid run -a discrete shift in number - drive it down, then drop it back down.. increase tax- property tax..
- [ ] see the size of the gap - what is the size of the gap.
- [ ] model inequality with shifted savings then do the same analysis
- [ ] (randomize - sensitivity shift)
- [ ] (spatially distributed density) - tenancy increase avg density - productivity in the city goes up somewhat.. get effect we're talking about.  rents are being extracted.. - savings level is actually declining if we do xyz. if the savings depended on just the share of work being captured by owners
- [ ] (mulitple replicate runs)
- [ ] Store metadata away from data so it is easier to keep if we need to delete large data sets
- [ ] Log file problem
- [ ] Look at effect of driving p_dot - p_dot on at x- value 
- [ ] Share of ownership to investors vs share of sales
- [ ] How to make a contour plot?
- [ ] How to see the gap between bidders - 
  - [ ] Display bid type (e.g. equity_limited etc)?
  - [ ] Show bids vs other expected proxy? 
  - [ ] store bid prices, reservation prices etc by type - how to best plot - just plot a vector-- or extract from agents? -- store for model - reservation_prices price/distance/property_id maybe  store pd.DataFrame(self.bids, columns=["time_step", "bid_price", "bid_distance", "property_id"]), "bids": lambda m: m.bids_dataframe()?
- [ ] Multi-replicate runs show up with error bars.
- [ ] Check: Implement individual_wealth_adjustment as stated in comment - Plot individual_wealth_adjustment etc
- [ ] Plots
  - [ ] Record/plot total rents
- [ ] Put grid displays in param sweep - at least for ones chosen - can choose any run/value to see output - to interogate state.


## Medium Priority

- [*] Track savings. Plots prices and saving. -  mean savings - distribution of savings. - 
- [*] Implement mortgages
- [ ] Check init_F
- [ ] Think about how to improve plots
  - [ ] Multiple replicate runs: Put likelihood bars e.g. on ownership share to show range. Larger city run overnight?
  - [ ] Record Workforce objects count of workers, newcomers, retiring as a sanity check - plot
  - [ ] Trace lines for a given land item as it folds over
  - [ ] Make a line for time for p_dot evolution
- [ ] Make a parameter for the variable 'm' in reservation price (and get_max_bid into a param?)  - max mortgage share 0.8  
- [ ] New Eqn for p_dot proxy (Add error checking to fail gracefully if no p_dot or no realized price data.)
- [ ] Try larger city


## Low Priority

- [ ] Add change_resident function like change_owner func
- [ ] Make function for worthwhile to work - Person and land can call for consistency
- [ ] Speed: If a property is listed for sale, calculate sale properties - calc warranted_rent etc. to speed up (either for all properties or when listed)
  - [ ] Record total run length for different changes. 
  - [ ] Try turning on off logging
  - [ ] Try memorize to speed up runs
- [ ] Fix test_auction for new plot? iPython notebook used some error messages I removed
- [ ] Remove redundant parameter sets to avoid changes in the wrong place


## DiR

- [ ] DiR has some kind of warning for depreciated syntax. Check.
- [ ] Consider why bids/realized_price is so far off warranted price (Old: Weird now warranted price is flat and realized not - or on top?)
  - [ ] Done? In bid calc realized_price: make price model use realized_price, around line 674
- [ ] Conversations: results experiments diagram measures figures. Talk through:
        self.rent_captured_by_finance  = 0 # TODO implement. make a marker for agents in the city
        self.share_captured_by_finance = 0 # TODO implement.
        self.urban_surplus   = 0 # TODO implement


### Experiments and plan (results experiments diagram measures figures)

-  [ ] Results: 
   -  [ ] Ownership pattern: under what circumstances do we see institutions take out ownership
   -  [ ] What circumstances allow to sustain a steady fraction of owner occupiers
   -  [ ] What share of rents are going to the bankers, share of generated surplus are the extracting.
   -  [ ] Share of ownership vs measure openness, threshold for entry. Distinction between people and with investors
-  [ ] Experiments:
   -  [ ] Differential incomes
   -  [ ] 2 Cities (one sucks it up)
   -  [ ] Density
   -  [ ] Feedbacks into urban surplus - extraction could change growth rate
   -  [ ] Shocks, resilience and hysteresis, cost to sell a home. 
   -  [ ] How to show resilience - fragility of homeowner regime, what exists beyond it
   -  [ ] Operating costs different for owners/not, discount rates may differ


## Consider for Future Work

- [ ] Do we only count amenity for workers, or those in the urban boundary?
- [ ] Add capital gains, tax first time buyer subsidy. Add transaction costs in WTP. It is the unfair floor that is the piece tht does the work.
- [ ] Add differential income
- [ ] How to do the resilience experiment?
- [ ] Size without population and/or density in different areas. 
- [ ] Calculate mean, median, or the percentage change over time, regression for direction and strength of the trends.
- [ ] Do investors need to be able to sell houses, do workers sell during their work lives?


## Orienting

- [ ] Search TODO
- [ ] Review TO SORT tasks below
- [ ] Trace flow - think about possible errors


## Check

- [x] Cases
      Case 1 no bid
      Case 2 investor wins
      Case 3 investor wins
      Case 4 with/without reservation price/with size of reservation price.
- [ ] Consider 0 warranted price - outside the city, any concern with transportation cost is only speculative - how to handle - return max(wage_premium - self.transport_cost + a * subsistence_wage, 0)
- [ ] How do we think about p_dot on the periphery?
- [ ] Why does max_bid not use transport_cost?
- [ ] Consider sigmoid transitions for get_bid transition
- [ ] Call little n firm size, remove N/F
- [ ] Make sure self.transport_cost_per_dist = self.params['c']
- [ ] Does changing c change anything?
 - [ ] plot people's ownership share not investor ownership share.