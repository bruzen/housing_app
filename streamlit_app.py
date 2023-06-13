import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from model import City  # Import the City model from model.py

def run_model(num_steps):
    # Create and run the model
    city = City()
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
