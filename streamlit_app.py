import random
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
from mesa import Model
from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation


class Cell:
    """
    Represents a single cell in Conway's Game of Life.
    """

    DEAD = 0
    ALIVE = 1

    def __init__(self, pos, model):
        self.pos = pos
        self.model = model
        self.state = self.DEAD

    def step(self):
        """
        Compute the next state of the cell based on its neighbors.
        """
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        live_neighbors = sum(1 for neighbor in neighbors if neighbor.state == self.ALIVE)

        if self.state == self.ALIVE:
            if live_neighbors < 2 or live_neighbors > 3:
                self.state = self.DEAD
        else:
            if live_neighbors == 3:
                self.state = self.ALIVE


class ConwaysGameOfLife(Model):
    """
    Represents the 2-dimensional array of cells in Conway's
    Game of Life.
    """

    def __init__(self, width=50, height=50):
        """
        Create a new playing area of (width, height) cells.
        """

        # Set up the grid and schedule
        self.grid = SingleGrid(width, height, torus=True)
        self.schedule = SimultaneousActivation(self)

        # Place a cell at each location, with some initialized to ALIVE and some to DEAD
        for x in range(width):
            for y in range(height):
                cell = Cell((x, y), self)
                if random.random() < 0.1:
                    cell.state = cell.ALIVE
                self.grid.place_agent(cell, (x, y))
                self.schedule.add(cell)

        self.running = True

    def step(self):
        """
        Have the scheduler advance each cell by one step
        """
        self.schedule.step()


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

    plt.imshow(grid_data, cmap='viridis')
    plt.axis('off')
    st.pyplot()


def main():
    st.title("Agent-Based Model Visualization")

    num_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=50)
    run_model(num_steps)


if __name__ == '__main__':
    main()
