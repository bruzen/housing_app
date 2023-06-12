import random
import numpy as np
import streamlit as st
import altair as alt


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

    # Convert data to numpy array
    agent_data = np.array(agent_data).T

    # Create a time series annotation plot
    time_series = np.arange(num_steps)
    annotations = np.argmax(agent_data, axis=0)

    data = np.concatenate((time_series.reshape(-1, 1), annotations.reshape(-1, 1)), axis=1)
    df = pd.DataFrame(data, columns=["Time", "Annotation"])

    line_chart = alt.Chart(df).mark_line().encode(
        x="Time",
        y="Annotation",
        tooltip=["Time", "Annotation"]
    )

    # Display the agent state evolution and the time series annotation plot using Streamlit
    st.line_chart(agent_data)
    st.altair_chart(line_chart, use_container_width=True)


def main():
    st.title("Agent-Based Model Visualization")

    num_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=50)
    run_model(num_steps)


if __name__ == "__main__":
    main()
