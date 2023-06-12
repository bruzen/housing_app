import random
import numpy as np
import streamlit as st
import altair as alt


class SimpleAgent:
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
        self.schedule = []

        # Create agents
        for i in range(self.num_agents):
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            agent = SimpleAgent(x, y, self.num_states)
            self.schedule.append(agent)

    def step(self):
        random.shuffle(self.schedule)
        for agent in self.schedule:
            agent.step(self)


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

    # Convert data to Pandas DataFrame for Altair visualization
    df = pd.DataFrame(agent_data, columns=[f"State {i}" for i in range(model.num_states)])
    df['Step'] = np.arange(num_steps)

    # Create the Altair time series plot
    chart = alt.Chart(df).transform_fold(
        [f"State {i}" for i in range(model.num_states)],
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


def main():
    st.title("Agent-Based Model Visualization")

    num_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=50)
    run_model(num_steps)


if __name__ == '__main__':
    main()
