# 📦 Housing Market Agent-Based Model

This Streamlit app provides a visualization of a housing market agent-based model focused on rent. The model simulates the behavior of agents in a housing market, where they make decisions regarding renting properties based on various factors such as price, location, and amenities.


## Installation Notes
### Clone the repository
git clone https://github.com/bruzen/housing_app

### Install requirements using Conda
conda create --name housing python=3.8
conda activate housing
conda install --file requirements.txt

where housing is the environment_name and https://github.com/bruzen/housing_app is the branch_name, in this case the url for the Github repository page

### Run the app
streamlit run streamlit_app.py

### Batch run
python batch_run.py

### Run model in Jupyter
jupyter notebook
Then click on analysis.ipynb in your web browser to open the file
(Or install nteract and click on the analysis.ipynb in Terminal)


## Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://housing-market-agent-model.streamlitapp.com/)

## Overview

The housing market agent-based model aims to capture the dynamics and interactions between agents in a simulated housing market. The model considers factors such as supply and demand, agent preferences, and market conditions to simulate the rental market.

## Features

- Interactive visualization of the housing market dynamics
- Display of agent behaviors and decision-making processes
- Real-time updates of model market indicators such as price, wage, and vacancy rate

## Usage

1. Adjust the parameters to set the initial market conditions, such as the number of agents, urban wage, and property availability.
2. Run the simulation to observe the behavior of agents and the changes in the housing market over time.
3. Analyze the visualizations and market indicators to gain insights into the dynamics of the housing market.
4. Experiment with different scenarios by modifying the parameters and running the simulation again.

## Further Reading

To learn more about agent-based modeling and its application in housing market simulations, consider exploring the following resources:

- Resource 1: [Introduction to Agent-Based Modeling]()
- Resource 2: [Agent-Based Models of Housing Markets: A Comprehensive Review]()
- Resource 3: [Agent-Based Modeling and Simulation in Housing Research]()
