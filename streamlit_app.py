import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from .model import ConwaysGameOfLife


def run_model(num_steps):
    # Create a Game of Life model
    model = ConwaysGameOfLife(width=50, height=50)

    # Create data for visualization
    agent_data = []

    # Run the model for the specified number of steps
    for step in range(num_steps):
        # Step the model
        model.step()

        # Collect agent state data
        agent_state_counts = np.zeros(2, dtype=int)
        for cell in model.schedule.agents:
            agent_state_counts[cell.state] += 1
        agent_data.append(agent_state_counts.copy())

    # Convert data to Pandas DataFrame for Altair visualization
    df = pd.DataFrame(agent_data, columns=['DEAD', 'ALIVE'])
    df['Step'] = np.arange(num_steps)

    # Create the Altair time series plot
    chart = alt.Chart(df).transform_fold(
        ['DEAD', 'ALIVE'],
        as_=['State', 'Count']
    ).mark_line().encode(
        x='Step:Q',
        y='Count:Q',
        color='State:N'
    ).properties(
        width=400,
        height=300
    )

    # Display the plots using Streamlit
    st.altair_chart(chart, use_container_width=True)

    # Display the agent state grid
    st.subheader("Agent State Grid")

    grid_data = np.zeros((model.grid.width, model.grid.height), dtype=int)
    for cell in model.schedule.agents:
        grid_data[cell.pos[0], cell.pos[1]] = cell.state

    fig, ax = plt.subplots()
    ax.imshow(grid_data, cmap='viridis')
    ax.axis('off')
    st.pyplot(fig)


def main():
    st.title("Agent-Based Model Visualization")

    num_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=50)
    run_model(num_steps)


if __name__ == '__main__':
    main()
