import streamlit as st
import matplotlib.pyplot as plt
from model import City

def run_model(subsistence_wage, working_periods, savings_rate, r_prime):
    # Create and run the model
    city = City(subsistence_wage=subsistence_wage, working_periods=working_periods,
                savings_rate=savings_rate, r_prime=r_prime)

    num_steps = 50  # Set the number of steps (you can modify this if needed)

    for t in range(num_steps):
        city.step()

    # Get output data
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
        subsistence_wage = st.slider("Subsistence Wage", min_value=30000., max_value=50000., value=40000., step=1000.)
        working_periods = st.slider("Working Periods", min_value=30, max_value=50, value=40)
        savings_rate = st.slider("Savings Rate", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
        r_prime = st.slider("R Prime", min_value=0.03, max_value=0.07, value=0.05, step=0.01)

    with col2:
        st.pyplot(fig)

def main():
    st.title("Agent-Based Model Visualization")

    subsistence_wage = st.slider("Subsistence Wage", min_value=30000., max_value=50000., value=40000., step=1000.)
    working_periods = st.slider("Working Periods", min_value=30, max_value=50, value=40)
    savings_rate = st.slider("Savings Rate", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    r_prime = st.slider("R Prime", min_value=0.03, max_value=0.07, value=0.05, step
