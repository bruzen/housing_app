import streamlit as st
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector

class SimpleAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.wealth = 1
    
    def step(self):
        self.move()
        self.interact()
    
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
    
    def interact(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if len(cellmates) > 1:
            other_agent = self.random.choice(cellmates)
            self.wealth += 1
            other_agent.wealth -= 1

class SimpleModel(Model):
    def __init__(self, N):
        self.num_agents = N
        self.schedule = RandomActivation(self)
        self.grid = SingleGrid(10, 10, torus=True)
        
        for i in range(self.num_agents):
            a = SimpleAgent(i, self)
            self.schedule.add(a)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.position_agent(a, (x, y))
        
        self.datacollector = DataCollector(
            model_reporters={"Wealth": lambda m: self.get_total_wealth()})
    
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
    
    def get_total_wealth(self):
        return sum([agent.wealth for agent in self.schedule.agents])

# Function to run the simulation
def run_simulation():
    model = SimpleModel(N=100)
    for _ in range(100):
        model.step()
    
    return model.datacollector.get_model_vars_dataframe()

# Streamlit App
def main():
    # Set page title and layout
    st.set_page_config(page_title="Agent-Based Model", layout="wide")
    
    # Set app title and description
    st.title("Agent-Based Model with Streamlit Cloud")
    st.write("This is a simple example of an agent-based model displayed using Streamlit Cloud.")
    
    # Run the simulation and retrieve data
    data = run_simulation()
    
    # Display data using Streamlit components
    st.header("Model Output")
    st.dataframe(data)
    
    st.header("Visualization")
    # Add visualization components here (e.g., matplotlib, plotly, etc.)
    # You can create plots or charts to visualize the model output.
    # For example, you can use st.pyplot() to display matplotlib plots.
    # Customize the visualization as per your specific needs.
    
# Run the app
if __name__ == '__main__':
    main()
