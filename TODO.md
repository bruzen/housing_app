
# TODO List

## Contributions:
1. First we put financialiation into a moded an agent based urban modoel that is well constrained and well understood, and show that in a plauzibly specified model, the financializaiton is driving the change in the classs structure. 
2. The economic litterature does not examine the class impact. Economists don't talk abouot. Socioloogists talk about it, but they don't model it.
3. We model boom bust dynamics/response. how it responds to chagnes in the outer world - fact you could get a change with an interest rate. what happens when the model is repeatedly struc. What are the impact effects and how do they play out?
4.  We ask the question about if there's a spillover to productivity. There's speculation about the link in the left win urban litterature but we haven't found any moodelling.

This is what's done in the climate litterature. The dynamics and resilience effects are not obvious from the equations and must be moodelled. The modoelling is important because empirical dynamics are really hard to establish. This is a 1 off world. We can't run 50 versions. You can, however moodel 50 versions. 

The basic fiancialization result is clear from looking at the model. The equations only make it obvious once you put it together, however. They way they are compbined is not obvsious.  It is important to note also that the results are not a consequence of building a model that will give any result, but only a consequence of building a model that behaves urban theory and the econoimc theory predicts. None of the equations were constructed to give a result. Each is our best about how it works. Each step was a best guess at how the process works and based on a knowlge of the litterature and the economics. The base theorly in neighter litterature clearly predicts the class effects (there is some work in the litturature e.g. Jacobs, reinvesting in Henry George litterauter- a few places where the economists have gone in that dirrection) We implemented a version of the best of the urban growth theories.

## Hypotheses
1. The financial sector affects the ownership of housing and the class structure of society, 
2. There are dynamic/resilience features of this model that make the effects worse
- Boom bust - pump wealth on/out of city on the boom and on the bust. 'There are sharks in the water' 1. Higher bid can amplify up swings - have to compete with speculators 2. can't get it back on down swing since outside finance offers a stable floor- buys up on way down (we expect larger effects as people are 1. displaced on the way up and then 2. evicted on the way down - can't use their spaces)
- Hysteresis - perturb, doesn't come back - e.g. interest rates go up.
- The way it aligns with long run changes in the landscape e.g. tech changing local info changes vulnerability to these shocks -- Depth of the basin changes - erodes systemic resilience - together these changes the capacity to hold value in landscape. (links to the productivity feedback - much will/can they invest in increasing their productivity/supporting kids/good food to grow brains. Education to increase productivity is the feature that makes productivity increases resilient to de-industrialization). This has implications for landscape/system/class structure.
3. This shift in ownership may have implications for urban productivity. (can actually displace productive uses - empty store fronts) - who can/will enter, how , who can rent spaces, speculative value may keep it empty (work spaces or living space - lowering pop), reduces consumption

## Model
- [ ] Imediately do experiments with the price of output - should give a decline in price of labour, may or may not result in changes in the housing market (record housing prices]
- [ ] Which interest rate is charged to investors
- [ ] Reduce demand for labour temporarily - reduce wages for labour - demand -- 
- [ ] Link finance and employment - vary price of output - vary the employment
- [*] Add cyclic structure with 1. local employment booms and 2. financial booms that are external. Examine the cases where they align and where they don't align

- [*] Investors/homeowners list home if best offer (e.g. offer from investor or newcomer with highest savings) is above reservation price. The newcomer comes into town. Is willing to go anywhere and works out the bid at any area. Agent goes around, having some idea about who might have a retiree. (Consider what happens if tenants can buy e.g. on their lifetime wealth trajectory, on captured rents for owners if there is turnover)

- [ ] Replace density - Vintage housing stock - houses live 60. added at edge. get bigger with time. get added after some time.. have a fixed density.. - replaced at cost 1. add at the edge and have a fixed density. 2. as time passes, density falls but size rises - the pattern in american cities - at the edges. justify by fidling with the transportation costs - transportation costs fall which let the city spreads faster (or fall as you faerthur out - lighway etc) 3. near the core do a replacement - replacement with high density. (every 60 years - agent cost.- depreciation curve for the house. -and a cost to go up - 6 story and make that work. then do that. then introduce) 4. then do another layer on the inside.
- [ ] (Spatially distributed density) - tenancy increase avg density - productivity in the city goes up somewhat.. get effect we're talking about.  rents are being extracted.. - savings level is actually declining if we do xyz. if the savings depended on just the share of work being captured by owners
- [ ] Check: Implement individual_wealth_adjustment as stated in comment - Plot individual_wealth_adjustment etc

## Ploting and runs
- [ ] Record/plot total rents
- [ ] Record housing wealth gap
- [ ] Show housing prices for batch run
- [ ] Record Workforce objects count of workers, newcomers, retiring as a sanity check - plot
- [ ] Trace lines for a given land item as it folds over
- [ ] Make a line for time for p_dot evolution
- [ ] Make a parameter for the variable 'm' in reservation price (and get_max_bid into a param?)  - max mortgage share 0.8  
- [ ] Try larger city
- [ ] Track savings. Plots prices and saving. -  mean savings - distribution of savings. - 
- [ ] Implement mortgages
- [ ] Check init_F

## Performance and Workflow
Speed/memory, plot and workflow to find parameters, run online - try storing the bid rent curve -- - what is the gap.. 

- [ ] Speed
  - [ ] Just keep housing - turn off storage..  cyclic pieces turn off the market. - turn off the production
  - [ ] Just plot bid rent curves and which is on top, and show bid rent curve relationships with parameters. See/store the size of the gap between bidders. Model inequality with different intitial savings level then do the same analysis of bids, who gets into the city, etc. Inequality will be hard since we don't close the loop. We can have wealth inequality but not income inequality easily.
  Display bid type (e.g. equity_limited etc)? store bid prices, reservation prices etc by type - how to best plot - just plot a vector-- or extract from agents? -- store for model - reservation_prices price/distance/property_id maybe  store pd.DataFrame(self.bids, columns=["time_step", "bid_price", "bid_distance", "property_id"]), "bids": lambda m: m.bids_dataframe()?
  - [ ] Look for cache/memory problems. Try pylint.
- [ ] Run model on the command line. See if that works on AWS.
- [ ] Param sweep for capital gains tax people/institutions .1 to .4  (or 0-1). Plot final ownership share, and ownership share every 10 steps - they fall at a different point .. what is the point of fall?
- [*] Perturb mid run - a discrete shift in number - drive it down, then drop it back down.. increase tax- property tax.. (maybe look at effect of driving p_dot)

- [ ] Sensitivity analysis
- [ ] Mulitple replicate runs
- [ ] Store metadata away from data so it is easier to keep if we need to delete large data sets
- [ ] Check log file for problems

## Low Priority
- [ ] Add change_resident function like change_owner func
- [ ] Make function for worthwhile to work - Person and land can call for consistency
- [ ] Speed: If a property is listed for sale, calculate sale properties - calc warranted_rent etc. to speed up (either for all properties or when listed)
  - [ ] Record total run length for different changes. 
  - [ ] Check if I can now use the default mesa scheduling code
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


Heat transfer = hA(T - Tambient), area is constant, say 1, Coffee temp  60 to 71 degrees Celsius
Person 
coefficient might be  10 to 50 W/(m²·C). for a person
hand maybe 36.1 to 37 degrees Celsius (97 to 98.6 degrees Fahrenheit a bit lower than core)
10*(60-36) to 50*(60-37) = 240 to 1150 watts (W)
Air
coefficent maybe 5 to 25 W/(m²·C) for air
maybe 20 Celsius for air (22 at heater 18 near window?)
5*(60-18) to 25*(60-22) = 210 to 950 W