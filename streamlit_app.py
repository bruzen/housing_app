import numpy as np
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import streamlit as st

# Define the Agent class
class MyAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True)  # Specify the 'moore' argument
        if neighbors:
            other_agent = self.random.choice(neighbors)
            self.model.grid.move_agent(self, other_agent.pos)

# Define the Model class
class MyModel(Model):
    def __init__(self, width, height, num_agents):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(self.width, self.height, torus=True)

        for i in range(self.num_agents):
            agent = MyAgent(i, self)
            self.schedule.add(agent)
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)

            try:
                self.grid.place_agent(agent, (x, y))
            except Exception as e:
                print(f"Error placing agent {agent.unique_id} at position ({x}, {y}): {e}")

        self.datacollector = DataCollector(model_reporters={"NumAgents": lambda m: m.schedule.get_agent_count()})

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

# Streamlit app
def run_model(num_steps):
    model = MyModel(10, 10, 100)
    for _ in range(num_steps):
        model.step()

    # Display the number of agents using Streamlit
    st.line_chart(model.datacollector.get_model_vars_dataframe())

# Run the Streamlit app
if __name__ == "__main__":
    num_steps = st.sidebar.slider("Number of Steps", 0, 100, 10)
    run_model(num_steps)
