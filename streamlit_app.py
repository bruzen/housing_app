import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from model import City  # Import the City model from model.py

def run_model(num_steps):
    # Create and run the model with the specified parameter values
    city = City(
        width=50,
        height=1,
        subsistence_wage=40000.,
        working_periods=40,
        savings_rate=0.3,
        r_prime=0.05,
        r_margin=0.01,
        prefactor=250,
        agglomeration_ratio=1.2,
        workers_share=0.72,
        property_tax_rate=0.04,
        mortgage_period=5.0,
        housing_services_share=0.3,
        maintenance_share=0.2,
        seed_population=0,
        density=100,
        max_mortgage_share=0.9,
        ability_to_carry_mortgage=0.28,
        wealth_sensitivity=0.1,
        wage_adjust_coeff_new_workers=0.5,
        wage_adjust_coeff_exist_workers=0.5,
        workforce_rural_firm=100,
        price_of_output=1.,
        beta_city=1.12,
        alpha_firm=0.18,
        z=0.5,
        init_wage_premium_ratio=0.2,
        init_city_extent=10.
    )
    for t in range(num_steps):
        city.step()

    # Get output data
    agent_out = city.datacollector.get_agent_vars_dataframe()
    model_out = city.datacollector.get_model_vars_dataframe()

    workers = model_out['workers'].to_numpy()
    wage = model_out['wage'].to_numpy()
    city_extent = model_out['city_extent'].to_numpy()

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
    st.pyplot(fig)

def main():
    st.title("Agent-Based Model Visualization")

    num_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=50)
    run_model(num_steps)

if __name__ == '__main__':
    main()
