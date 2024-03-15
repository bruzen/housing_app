default_parameters = {
    'run_notes': 'Exploring behavior.',
    'subfolder': None,
    'width':     50, #30,
    'height':    50, #30,

    # INTERVENTIONS
    'interventions_on': False,
    # 'interventions_on': True,
    'interventions':  {
        # 'Output price down': {'var': 'firm.price_of_output', 'val':  8, 'time': 25},
        # 'Output price back':   {'var': 'firm.price_of_output', 'val': 10, 'time': 45},
    #     # 'Person capital gains tax down': {'var': 'person.capital_gains_tax', 'val':  .1, 'time': 25},
    #     # 'Person capital gains tax up':   {'var': 'person.capital_gains_tax', 'val': .2, 'time': 45},
    #     # 'Investor capital gains tax down': {'var': 'investor.capital_gains_tax', 'val':  .1, 'time': 25},
    #     # 'Investor capital gains tax up':   {'var': 'investor.capital_gains_tax', 'val': .2, 'time': 45},
    #     # 'investor_turnover up': {'var': 'investor.investor_turnover', 'val':  .5, 'time': 10},
    #     # 'investor_turnover down':   {'var': 'investor.investor_turnover', 'val': .05, 'time': 20},
    #     # Add more interventions...
    },

    # FLAGS
    'demographics_on': True,  # Set flag to False for debugging to check firm behaviour without demographics or housing market
    'center_city':     False, # Flag for city center in center if True, or bottom corner if False
    # 'random_init_age': False,  # Flag for randomizing initial age. If False, all workers begin at age 0
    'random_init_age': True,  # Flag for randomizing initial age. If False, all workers begin at age 0

    # DATA STORAGE
    'store_agent_data': True,
    'no_decimals':      1,

    # STORAGE VALUES, JUST FOR MODEL FAST
    'distances': None,
    'newcomer_savings': None,

    # LABOUR MARKET AND FIRM PARAMETERS
    'subsistence_wage': 40000.,     # psi wage fo rural worker
    'seed_population':  400,        # 
    'dist':   1, 
    'init_wage_premium_ratio': 0.2, # 1.2,
    'init_city_extent': 10.,        # initial city extent
    'init_F': 100.0,                # initial number  of firms
    'init_k': 50000.0,              # initial firm capital stock 
    'init_n': 10,                   # initial firm size (number of workers)

    # PARAMETERS MOST LIKELY TO AFFECT SCALE
    'c': 500.0,                     # transportation costs
    'price_of_output': 10,          # received by firms per unit of output
    # 'A'                           # Scale parameter  in production function. Our central value is 1800
    'A_productivity_link': False,   # Flag for link between productivity and A. Eqn is A_base + (1-share) * A_slope, where share = 0 if False CHECK
    #       "share" is share of rents capture by investors 
    'A_base': 1200,                 # Fixed component of A
    'A_slope':600,                  # Part of A from local investment 
    'density': 100,                 # nmber of workers peer unit land
    'alpha': 0.18,                  # Cobb-Douglas exponent for firm capital
    'beta':  0.75,                  # Cobb-Douglas exponent for firm labour
    'gamma': 0.003,                 # exponent for agglomeration effect of adjusted city population N
    'overhead': .5,                 # extra cost associated with an additional employee 
    'mult': 1.2,                    # ratiio of population affecting agglomeration to total number of workers 

    # SPEED OF ADJUSTMENT PARAMETERS 
    'adjN': 0.015,                  # Adjustment speed firm
    'adjk': 0.010,                  # Adjustment speed firm capital
    'adjn': 0.075,                  # Adjustment speed firm jworkforce
    'adjF': 0.3,                    # Adjustment speed number of firms
    'adjw': 0.002,                  # Adjustment speed wage paid by firm
    'adjs': 0.2,                    # Adjustment speed aggregate worker supply 
    'adjd': 0.2,                    # Adjustment speed aggregate worker Demand 
    'adjp': 0.2,                    # Adjustment speed aggregate agglom_pop -Population

    # HOUSING AND MORTGAGE MARKET PARAMETERS
    'r_prime': 0.05,               # Bank rate, basic interst rate
    'r_margin': 0.01,              # additional return bank requires to lend
    'r_investor': 0.05,            # Next best alternative return for investor
    'ability_to_carry_mortgage': 0.28, # share of total income that may be spent on mortgage
    'wealth_sensitivity': 0.1,     # degree of mortgag3e tightness for low wealth
    'mortgage_period': 5.0,        # T, in years
    'working_periods': 40,         # years in workforce before retirement
    'savings_rate': 0.3,           # share of income homeowners save
    'discount_rate': 0.07,         # rate at which homeowners discount future income
    'housing_services_share': 0.3, # a fraction of rural subsistence spent on housing
    'maintenance_share': 0.2,      # b annual fraction of building cost spent on  maintenance
    'property_tax_rate': 0.04,     # tau, annual rate, was c
    'max_mortgage_share': 0.9,     # m share of property price elligible for mortgage
    'cg_tax_per':   0.01,          # Capital gains  tax rate for owner-occupiers  (0-1)
    'cg_tax_invest': 0.15,         # Capital gains  tax rate for investors  (0-1)
    'investor_turnover': 0.05,     # fraction of investor land holdings put up for sale each period
    'investor_expectations': 1.,   # "animal spirits," optimism ofinvestor estimate of capital gains (near 1)
}