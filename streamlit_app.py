import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt

from model.model import City

def run_model(num_steps, subsistence_wage, working_periods, savings_rate, r_prime):
    width = 50
    height = 1


    # parameters = {
    #     'width': 50,
    #     'height': 1,
    #     'init_city_extent': 10.,
    #     'seed_population': 10,
    #     'density': 100,
    #     'subsistence_wage': 40000.,
    #     'init_wage_premium_ratio': 0.2,
    #     'workforce_rural_firm': 100,
    #     'price_of_output': 1.,
    #     'alpha_F': 0.18,
    #     'beta_F': 0.72,
    #     'beta_city': 1.12,
    #     'gamma': 0.02,
    #     'Z': 0.5,
    #     'firm_adjustment_parameter': 0.25,
    #     'wage_adjustment_parameter': 0.5,
    #     'mortgage_period': 5.0,
    #     'working_periods': 40,
    #     'savings_rate': 0.3,
    #     'r_prime': 0.05,
    #     'r_margin': 0.01,
    #     'property_tax_rate': 0.04,
    #     'housing_services_share': 0.3,
    #     'maintenance_share': 0.2,
    #     'max_mortgage_share': 0.9,
    #     'ability_to_carry_mortgage': 0.28,
    #     'wealth_sensitivity': 0.1,
    # }

    # city = City(parameters)

    # Create and run the model
    city = City(width=width, height=height, subsistence_wage=subsistence_wage, working_periods=working_periods,
                savings_rate=savings_rate, r_prime=r_prime)

    for t in range(num_steps):
        city.step()

    # Get output data
    model_out = city.datacollector.get_model_vars_dataframe()
    workers = np.array(model_out['workers'])
    wage = np.array(model_out['wage'])
    city_extent = np.array(model_out['city_extent'])
    time = np.arange(num_steps)

    # Set up the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Model Output', fontsize=16)

    # Plot 1: Workers vs Time
    axes[0, 0].plot(time, workers, label='Workers', color='blue')
    axes[0, 0].plot(time, wage, label='Wage', linestyle='--', color='red')
    #  axes[0, 0].plot(time, city_extent, label='City Extent', linestyle='dotted', color='red')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Number of Workers')
    # axes[0, 0].set_title('Workers vs Wage')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True)

  # Plot 2: Wage vs Time
    axes[0, 0].plot(time, workers, label='Workers', color='blue')
    axes[0, 0].plot(time, wage, label='Wage', linestyle='--', color='green')
    axes[0, 0].plot(time, city_extent, label='City Extent', linestyle='dotted', color='red')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Number')
    axes[0, 0].set_title('Workers vs Wage')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True)
    
    # Plot 3: Wage vs Workers
    axes[0, 1].plot(workers, wage, color='purple')
    axes[0, 1].set_title('Subplot 2')
    axes[0, 1].set_xlabel('Workers')
    axes[0, 1].set_ylabel('Wage')
    axes[0, 1].grid(True)

    # Plot 4: City Extent vs time
    axes[1, 0].plot(city_extent, time, color='magenta')
    axes[1, 0].set_title('Subplot 3')
    axes[1, 0].set_xlabel('Workers')
    axes[1, 0].set_ylabel('City Extent')
    axes[1, 0].grid(True)

    # Plot 5: Workers vs Wage (subplot 4)
    axes[1, 1].plot(workers, wage, color='brown')
    axes[1, 1].set_title('Subplot 4')
    axes[1, 1].set_xlabel('Workers')
    axes[1, 1].set_ylabel('Wage')
    axes[1, 1].grid(True)

    plt.tight_layout()

    # Display the plots using Streamlit
    st.title("Housing Market Model Output")
    st.pyplot(fig)

    st.markdown("---")

    st.header("Explore Existing Data")
    st.pyplot(fig)


def main():
    num_steps = st.sidebar.slider("Number of Steps", min_value=1, max_value=100, value=50)
    subsistence_wage = st.sidebar.slider("Subsistence Wage", min_value=30000., max_value=50000., value=40000., step=1000.)
    working_periods = st.sidebar.slider("Working Periods", min_value=30, max_value=50, value=40)
    savings_rate = st.sidebar.slider("Savings Rate", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    r_prime = st.sidebar.slider("R Prime", min_value=0.03, max_value=0.07, value=0.05, step=0.01)

    run_model(num_steps, subsistence_wage, working_periods, savings_rate, r_prime)

if __name__ == "__main__":
    main()
