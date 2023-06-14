import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from model import City  # Import the City model from model.py

def run_model(num_steps, subsistence_wage, working_periods, savings_rate, r_prime):
    width = 50  # Set default values for width and height
    height = 1

    # Create and run the model
    city = City(width=width, height=height, subsistence_wage=subsistence_wage, working_periods=working_periods,
                savings_rate=savings_rate, r_prime=r_prime)

    for t in range(num_steps):
        city.step()

    # Get output data
    agent_out = city.datacollector.get_agent_vars_dataframe()
    model_out = city.datacollector.get_model_vars_dataframe()

    workers = np.array(model_out.workers)
    wage = np.array(model_out.wage)
    city_extent = np.array(model_out.city_extent)

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
    col1, col2 = st.columns(2)

    with col1:
        st.title("Agent-Based Model Visualization")
        st.slider("Number of Steps", min_value=1, max_value=100, value=num_steps, key="num_steps")
        st.slider("Subsistence Wage", min_value=30000., max_value=50000., value=subsistence_wage, step=1000., key="subsistence_wage")
        st.slider("Working Periods", min_value=30, max_value=50, value=working_periods, key="working_periods")
        st.slider("Savings Rate", min_value=0.1, max_value=0.5, value=savings_rate, step=0.05, key="savings_rate")
        st.slider("R Prime", min_value=0.03, max_value=0.07, value=r_prime, step=0.01, key="r_prime")
    
    with col2:
        st.pyplot(fig)

def main():
    st.title("Agent-Based Model Visualization")

    num_steps = 50  # Default value for num_steps
    subsistence_wage = 40000.  # Default value for subsistence_wage
    working_periods = 40  # Default value for working_periods
    savings_rate = 0.3  # Default value for savings_rate
    r_prime = 0.05  # Default value for r_prime

    run_model(num_steps, subsistence_wage, working_periods, savings_rate, r_prime)

if __name__ == "__main__":
    main()
