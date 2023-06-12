import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import pandas as pd


class Agent:
    def __init__(self, x, y, num_states):
        self.x = x
        self.y = y
        self.state = random.randint(0, num_states - 1)

    def step(self, model):
        self.state += 1
        self.state %= model.num_states


class SimpleModel:
    def __init__(self, N, width, height, num_states):
        self.num_agents = N
        self.grid_width = width
        self.grid_height = height
        self.num_states = num_states
        self.grid = np.zeros((width, height), dtype=int)
        self.schedule = []

        # Create agents
        for i in range(self.num_agents):
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            agent = Agent(x, y, self.num_states)
            self.schedule.append(agent)
            self.grid[x, y] = agent.state

    def step(self):
        random.shuffle(self.schedule)
        for agent in self.schedule:
            agent.step(self)
            self.grid[agent.x, agent.y] = agent.state


def run_model(num_steps):
    # Create a simple model
    model = SimpleModel(N=100, width=10, height=10, num_states=5)

    # Create data for visualization
    agent_data = []

    # Run the model for the specified number of steps
    for step in range(num_steps):
        # Step the model
        model.step()

        # Collect agent state data
        agent_state_counts = np.zeros(model.num_states, dtype=int)
        for agent in model.schedule:
            agent_state_counts[agent.state] += 1
        agent_data.append(agent_state_counts.copy())

    # Convert data to numpy array and transpose for plotting
    agent_data = np.array(agent_data).T

    # Plot grid and agent state evolution
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.imshow(model.grid, cmap='viridis')
    ax1.set_title("Agent State Grid")
    ax1.axis('off')

    ax2.plot(agent_data)
    ax2.set_title("Agent State Evolution")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Count")

    # Convert agent data to dataframe for Altair plot
    df = pd.DataFrame(agent_data.T, columns=[f"State {i}" for i in range(model.num_states)])
    df["Step"] = range(num_steps)

    # Create the Altair line plot with annotations
    chart = alt.Chart(df).mark_line().encode(
        x="Step",
        y=[f"State {i}" for i in range(model.num_states)],
        color=alt.Color("state:N", legend=None)
    ).properties(
        width=500,
        height=300
    ).interactive()

    # Add text annotations to the plot
    text = chart.mark_text(
        align='left',
        baseline='middle',
        dx=5,
        dy=-5,
        color='black'
    ).encode(
        text=alt.Text("value:Q", format=".0f"),
        opacity=alt.value(0.6)
    )

    annotated_chart = chart + text

    ax3.set_title("Agent State Evolution with Annotations")
    ax3.axis('off')
    st.altair_chart(annotated_chart, use_container_width=True)

    # Show the plot in Streamlit
    st.pyplot(fig)


# Run the model for 50 steps
run_model(50)
