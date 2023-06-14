import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from model import City

def run_model(num_steps, subsistence_wage, working_periods, savings_rate, r_prime):
    width = 50
    height = 1

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

    # Create the plots using Altair
    data = pd.DataFrame({'Time': time, 'Workers': workers, 'Wage': wage, 'City Extent': city_extent})

    chart1 = alt.Chart(data).mark_line().encode(
        x='Time',
        y='Workers',
        color=alt.value('blue'),
        tooltip=['Time', 'Workers']
    ).properties(
        title='Workers over Time',
        width=400,
        height=300
    )

    chart2 = alt.Chart(data).mark_line().encode(
        x='Time',
        y='Wage',
        color=alt.value('green'),
        tooltip=['Time', 'Wage']
    ).properties(
        title='Wage over Time',
        width=400,
        height=300
    )

    chart3 = alt.Chart(data).mark_line().encode(
        x='Time',
        y='City Extent',
        color=alt.value('red'),
        tooltip=['Time', 'City Extent']
    ).properties(
        title='City Extent over Time',
        width=400,
        height=300
    )

    # Combine the plots into a single column
    plots = chart1 | chart2 | chart3

    # Display the plots using Streamlit
    st.title("Agent-Based Model Visualization")
    st.altair_chart(plots, use_container_width=True)

def main():
    num_steps = st.sidebar.slider("Number of Steps", min_value=1, max_value=100, value=50)
    subsistence_wage = st.sidebar.slider("Subsistence Wage", min_value=30000., max_value=50000., value=40000., step=1000.)
    working_periods = st.sidebar.slider("Working Periods", min_value=30, max_value=50, value=40)
    savings_rate = st.sidebar.slider("Savings Rate", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    r_prime = st.sidebar.slider("R Prime", min_value=0.03, max_value=0.07, value=0.05, step=0.01)

    run_model(num_steps, subsistence_wage, working_periods, savings_rate, r_prime)

if __name__ == "__main__":
    main()
