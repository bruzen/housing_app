# TODO List
## High Priority

- [ ] Param sweep line graphs with colors for values of params - save to photo.. for later ones.. and then arrange as a grid
- [ ] How to see the gap to win bid - between bidders - 
  - [ ] Display bid type (e.g. equity_limited etc)?
  - [ ] Show bids vs other expected proxy? 
  - [ ] store bid prices, reservation prices etc by type - how to best plot - just plot a vector-- or extract from agents? -- store for model - reservation_prices price/distance/property_id maybe  store pd.DataFrame(self.bids, columns=["time_step", "bid_price", "bid_distance", "property_id"]), "bids": lambda m: m.bids_dataframe()?
- [ ] Check: Implement individual_wealth_adjustment as stated in comment - Plot individual_wealth_adjustment etc
- [ ] Plots
  - [ ] Record/plot total rents
  - [ ] Urban wage and MPL
  - [ ] Check names and labels on line plots

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

## Low Priority

- [ ] Add change_resident function like change_owner func
- [ ] Make function for worthwhile to work - Person and land can call for consistency
- [ ] Speed: If a property is listed for sale, calculate sale properties - calc warranted_rent etc. to speed up (either for all properties or when listed)
- [ ] Fix test_auction for new plot? iPython notebook used some error messages I removed


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