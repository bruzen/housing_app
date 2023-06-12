import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import pandas as pd
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer

# Define the Agent class
class MyAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.state = np.random.choice(['A', 'B', 'C'])

    def step(self):
        # The agent randomly changes its state
        self.state = np.random.choice(['A', 'B', 'C'])

# Define the Model class
class MyModel(Model):
    def __init__(self, N):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.grid = MultiGrid(10, 10, torus=True)
        self.datacollector = DataCollector(
            model_reporters={"Count_A": lambda m: self.count_agents(m, 'A'),
                             "Count_B": lambda m: self.count_agents(m, 'B'),
                             "Count_C": lambda m: self.count_agents(m, 'C')}
        )
        
        # Create agents
        for i in range(self.num_agents):
            agent = MyAgent(i, self)
            self.schedule.add(agent)

            # Place agents on the grid
            x = np.random.randint(0, self.grid.width)
            y = np.random.randint(0, self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.running = True

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()

    def count_agents(self, model, state):
        agents = [agent for agent in model.schedule.agents if agent.state == state]
        return len(agents)

# Create the Altair time series plot
def plot_aggregate_state(model):
    df = model.datacollector.get_model_vars_dataframe()
    df = df.reset_index().melt('Index', var_name='State', value_name='Count')
    
    chart = alt.Chart(df).mark_line().encode(
        x='Index:Q',
        y='Count:Q',
        color='State:N'
    ).properties(
        width=400,
        height=300
    )
    
    return chart

# Create the canvas grid visualization
def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "r": 0.5}
    if agent.state == 'A':
        portrayal["Color"] = "red"
    elif agent.state == 'B':
        portrayal["Color"] = "green"
    elif agent.state == 'C':
        portrayal["Color"] = "blue"
    return portrayal

# Set up the visualization server
grid = CanvasGrid(agent_portrayal, 10, 10, 500, 500)
chart = plot_aggregate_state

server = ModularServer(
    MyModel,
    [grid, chart],
    "My Model",
    {"N": 100}  # Number of agents
)
server.port = 8521  # Set the port for the server
server.launch()
