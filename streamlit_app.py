import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from model import City  # Import the City model from model.py

def run_model(num_steps, width, height, subsistence_wage, working_periods, savings_rate, r_prime, r_margin,
              prefactor, agglomeration_ratio, workers_share, property_tax_rate, mortgage_period,
              housing_services_share, maintenance_share, seed_population, density, max_mortgage_share,
              ability_to_carry_mortgage, wealth_sensitivity, wage_adjust_coeff_new_workers,
              wage_adjust_coeff_exist_workers, workforce_rural_firm, price_of_output, beta_city,
              alpha_firm, z, init_wage_premium_ratio, init_city_extent):
    # Create and run the model
    city = City(width=width, height=height, subsistence_wage=subsistence_wage, working_periods=working_periods,
                savings_rate=savings_rate, r_prime=r_prime, r_margin=r_margin, prefactor=prefactor,
                agglomeration_ratio=agglomeration_ratio, workers_share=workers_share,
                property_tax_rate=property_tax_rate, mortgage_period=mortgage_period,
                housing_services_share=housing_services_share, maintenance_share=maintenance_share,
                seed_population=seed_population, density=density, max_mortgage_share=max_mortgage_share,
                ability_to_carry_mortgage=ability_to_carry_mortgage, wealth_sensitivity=wealth_sensitivity,
                wage_adjust_coeff_new_workers=wage_adjust_coeff_new_workers,
                wage_adjust_coeff_exist_workers=wage_adjust_coeff_exist_workers,
                workforce_rural_firm=workforce_rural_firm, price_of_output=price_of_output, beta_city=beta_city,
                alpha_firm=alpha_firm, z=z, init_wage_premium_ratio=init_wage_premium_ratio,
                init_city_extent=init_city_extent)

    for t in range(num_steps):
        city.step()

    # Get output data
    agent_out = city.datacollector.get_agent_vars_dataframe()
    model_out = city.datacollector.get_model_vars_dataframe()

    workers = model_out.workers
    wage = model_out.wage
    city_extent = model_out.city_extent

    # Create the plots
    fig, ax = plt.subplots(2, 2)
    fig.tight_layout(h_pad=4)
    fig.suptitle('Model Output')
    plt.subplots_adjust(top=0.85)

    ax[0, 0].plot(workers, label='workers')
    ax[0, 0].plot(wage, linestyle='--', label='wage')
    ax[0, 0].plot(city_extent, linestyle='dotted', label='extent')
    ax[0, 0].set_title('Workers vs Wage')
    ax[0, 0].set_xlabel('Time')
    ax[0, 0].set_ylabel('Number')
    ax[0, 0].legend(loc='upper right')

    ax[0, 1].plot(workers, wage)
    ax[0, 1].plot(city_extent, wage)
    ax[0, 1].set_title('subplot 2')

    ax[1, 0].plot(workers, wage)
    ax[1, 0].set_title('subplot 3')
    ax[1, 0].set_xlabel('workers')
    ax[1, 0].set_ylabel('wage')

    ax[1, 1].plot(workers, wage)
    ax[1, 1].set_title('subplot 4')

    # Display the plots using Streamlit
    col1, col2 = st.beta_columns(2)

    with col1:
        st.title("Agent-Based Model Visualization")
        num_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=50)
        width = st.slider("Width", min_value=10, max_value=100, value=50)
        height = st.slider("Height", min_value=1, max_value=10, value=1)
        subsistence_wage = st.slider("Subsistence Wage", min_value=30000., max_value=50000., value=40000., step=1000.)
        working_periods = st.slider("Working Periods", min_value=30, max_value=50, value=40)
        savings_rate = st.slider("Savings Rate", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
        r_prime = st.slider("R Prime", min_value=0.03, max_value=0.07, value=0.05, step=0.01)
        r_margin = st.slider("R Margin", min_value=0.005, max_value=0.02, value=0.01, step=0.001)
        prefactor = st.slider("Prefactor", min_value=200, max_value=300, value=250)
        agglomeration_ratio = st.slider("Agglomeration Ratio", min_value=1.0, max_value=1.5, value=1.2, step=0.1)
        workers_share = st.slider("Workers Share", min_value=0.5, max_value=1.0, value=0.72, step=0.05)
        property_tax_rate = st.slider("Property Tax Rate", min_value=0.02, max_value=0.06, value=0.04, step=0.01)
        mortgage_period = st.slider("Mortgage Period", min_value=3.0, max_value=7.0, value=5.0, step=0.5)
        housing_services_share = st.slider("Housing Services Share", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
        maintenance_share = st.slider("Maintenance Share", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        seed_population = st.slider("Seed Population", min_value=0, max_value=100, value=0)
        density = st.slider("Density", min_value=50, max_value=150, value=100)
        max_mortgage_share = st.slider("Max Mortgage Share", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
        ability_to_carry_mortgage = st.slider("Ability to Carry Mortgage", min_value=0.1, max_value=0.5, value=0.28, step=0.05)
        wealth_sensitivity = st.slider("Wealth Sensitivity", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
        wage_adjust_coeff_new_workers = st.slider("Wage Adjust Coeff for New Workers", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        wage_adjust_coeff_exist_workers = st.slider("Wage Adjust Coeff for Existing Workers", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        workforce_rural_firm = st.slider("Workforce Rural Firm", min_value=50, max_value=150, value=100)
        price_of_output = st.slider("Price of Output", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
        beta_city = st.slider("Beta City", min_value=1.0, max_value=1.5, value=1.12, step=0.05)
        alpha_firm = st.slider("Alpha Firm", min_value=0.1, max_value=0.3, value=0.18, step=0.01)
        z = st.slider("Z", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
        init_wage_premium_ratio = st.slider("Initial Wage Premium Ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
        init_city_extent = st.slider("Initial City Extent", min_value=5.0, max_value=20.0, value=10.0, step=1.0)

    with col2:
        st.pyplot(fig)

def main():
    st.title("Agent-Based Model Visualization")

    num_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=50)
    width = st.slider("Width", min_value=10, max_value=100, value=50)
    height = st.slider("Height", min_value=1, max_value=10, value=1)
    subsistence_wage = st.slider("Subsistence Wage", min_value=30000., max_value=50000., value=40000., step=1000.)
    working_periods = st.slider("Working Periods", min_value=30, max_value=50, value=40)
    savings_rate = st.slider("Savings Rate", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    r_prime = st.slider("R Prime", min_value=0.03, max_value=0.07, value=0.05, step=0.01)
    r_margin = st.slider("R Margin", min_value=0.005, max_value=0.02, value=0.01, step=0.001)
    prefactor = st.slider("Prefactor", min_value=200, max_value=300, value=250)
    agglomeration_ratio = st.slider("Agglomeration Ratio", min_value=1.0, max_value=1.5, value=1.2, step=0.1)
    workers_share = st.slider("Workers Share", min_value=0.5, max_value=1.0, value=0.72, step=0.05)
    property_tax_rate = st.slider("Property Tax Rate", min_value=0.02, max_value=0.06, value=0.04, step=0.01)
    mortgage_period = st.slider("Mortgage Period", min_value=3.0, max_value=7.0, value=5.0, step=0.5)
    housing_services_share = st.slider("Housing Services Share", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    maintenance_share = st.slider("Maintenance Share", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    seed_population = st.slider("Seed Population", min_value=0, max_value=100, value=0)
    density = st.slider("Density", min_value=50, max_value=150, value=100)
    max_mortgage_share = st.slider("Max Mortgage Share", min_value=0.5, max_value=1.0, value=0.9, step=0.05)
    ability_to_carry_mortgage = st.slider("Ability to Carry Mortgage", min_value=0.1, max_value=0.5, value=0.28, step=0.05)
    wealth_sensitivity = st.slider("Wealth Sensitivity", min_value=0.05, max_value=0.5, value=0.1, step=0.05)
    wage_adjust_coeff_new_workers = st.slider("Wage Adjust Coeff for New Workers", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    wage_adjust_coeff_exist_workers = st.slider("Wage Adjust Coeff for Existing Workers", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    workforce_rural_firm = st.slider("Workforce Rural Firm", min_value=50, max_value=150, value=100)
    price_of_output = st.slider("Price of Output", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
    beta_city = st.slider("Beta City", min_value=1.0, max_value=1.5, value=1.12, step=0.05)
    alpha_firm = st.slider("Alpha Firm", min_value=0.1, max_value=0.3, value=0.18, step=0.01)
    z = st.slider("Z", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    init_wage_premium_ratio = st.slider("Initial Wage Premium Ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
    init_city_extent = st.slider("Initial City Extent", min_value=5.0, max_value=20.0, value=10.0, step=1.0)

    run_model(num_steps, width, height, subsistence_wage, working_periods, savings_rate, r_prime, r_margin,
              prefactor, agglomeration_ratio, workers_share, property_tax_rate, mortgage_period,
              housing_services_share, maintenance_share, seed_population, density, max_mortgage_share,
              ability_to_carry_mortgage, wealth_sensitivity, wage_adjust_coeff_new_workers,
              wage_adjust_coeff_exist_workers, workforce_rural_firm, price_of_output, beta_city,
              alpha_firm, z, init_wage_premium_ratio, init_city_extent)

if __name__ == "__main__":
    main()
