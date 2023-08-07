import os
import subprocess
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from   plotly.subplots import make_subplots

from model.model import City

@st.cache_data()
def run_simulation(num_steps, parameters):
    city = City(num_steps, **parameters)
    city.run_model()

    # Get output data
    agent_out = city.datacollector.get_agent_vars_dataframe()
    model_out = city.datacollector.get_model_vars_dataframe()
    return agent_out, model_out

@st.cache_data()
def plot_agent_heatmap(df, selected_variable):
    # Define the color scale limits based on the minimum and maximum value of the selected variable
    z_min = df[selected_variable].min()
    z_max = df[selected_variable].max()

    # Create a list of figures for each time step
    figs = []
    for time_step in df['time_step'].unique():
        temp_df = df[df['time_step'] == time_step]
        hover_text = f"{selected_variable}: " + temp_df[selected_variable].astype(str) + '<br>ID: ' + temp_df['id'].astype(str)
        fig = go.Figure(data=go.Heatmap(
            z=temp_df[selected_variable],
            x=temp_df['x'],
            y=temp_df['y'],
            hovertext=hover_text,
            colorscale='viridis',
            zmin=z_min,
            zmax=z_max,
            colorbar=dict(title=selected_variable, titleside='right')
        ))
        fig.update_layout(title=f'Time Step: {time_step}', xaxis_nticks=20, yaxis_nticks=20)
        figs.append(fig)

    # Use subplot to add a slider through each time step
    final_fig = make_subplots(rows=1, cols=1)

    # Add traces from each figure to the final figure
    for i, fig in enumerate(figs, start=1):
        for trace in fig.data:
            final_fig.add_trace(
                go.Heatmap(
                    z=trace['z'],
                    x=trace['x'],
                    y=trace['y'],
                    hovertext=trace['hovertext'],
                    colorscale=trace['colorscale'],
                    zmin=z_min,
                    zmax=z_max,
                    colorbar=trace.colorbar,
                    visible=(i==1)  # only the first trace is visible
                ),
                row=1,
                col=1  # add the trace to the first subplot
            )

    # Create frames for each time step
    final_fig.frames = [go.Frame(data=[figs[i].data[0]], name=str(i)) for i in range(len(figs))]

    # Create a slider to navigate through each time step
    steps = [dict(label=str(i), method="animate", args=[[str(i)], dict(frame=dict(duration=300, redraw=True))]) for i in range(len(figs))]
    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    final_fig.update_layout(height=600, width=800, sliders=sliders)
    # final_fig.update_layout(height=600, width=800, title_text=f"{selected_variable} Heatmap Over Time Steps", sliders=sliders)

    # Show the plot in Streamlit
    st.plotly_chart(final_fig)

@st.cache_data()
def plot_model_data(model_out):
    workers = np.array(model_out['workers'])
    wage = np.array(model_out['wage'])
    city_extent_calc = np.array(model_out['city_extent_calc'])
    time = np.arange(len(workers))

    # Set up the figure and axes
    fig, axes = plt.subplots(3, 2, figsize=(10, 15))
    fig.suptitle('Model Output', fontsize=16)
    
    # New plot 1L: evolution of the wage  
    axes[0, 0].plot(time, wage, color='red')
    axes[0, 0].set_title('City Extent and Wage (Rises)')
    axes[0, 0].set_title('Evolution of the wage ')  
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Wage')
    axes[0, 0].grid(True)

    # New plot 3L: evolution of the city extent = l?
    axes[0,1].plot(time, city_extent_calc, color='red')
    axes[0,1].set_title('Evolution of the City Extent (Rises)')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('City Extent')
    axes[0,1].grid(True)

    # New plot 2R:  evolution of the workforce
    axes[1, 0].plot(time, workers, color='purple') 
    axes[1, 0].set_title('Evolution of the Workforce (Rises)')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Workers')
    axes[1, 0].grid(True)

    # Plot 2L: city extent and workforce  
    axes[1, 1].plot(city_extent_calc, workers, color='magenta')
    axes[1, 1].set_title('City Extent and Workforce (Curves Up)')
    axes[1, 1].set_xlabel('City Extent')
    axes[1, 1].set_ylabel('Workers')
    axes[1, 1].grid(True)              
    
    # New plot 3L: city extent and wage
    axes[2, 0].plot(time, city_extent_calc, color='red')
    # axes[2, 0].set_title('City Extent and Wage (Linear)')
    axes[2, 0].set_title('City Extent and Wage (Curves Up)')
    axes[2, 0].set_xlabel('Wage')
    axes[2, 0].set_ylabel('City Extent')
    axes[2, 0].grid(True)

    # New plot 1R: workforce response to wage
    axes[2, 1].plot(wage, workers, color='purple')
    axes[2, 1].set_title('Workforce Response to Wage')
    axes[2, 1].set_xlabel('Wage')
    axes[2, 1].set_ylabel('Workers')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

def plot_batch_run_data():
    batch_run_folders = get_batch_run_folders()
    selected_folder = st.selectbox("Select Batch Run Folder", batch_run_folders)
    folder_path = os.path.join("output_data", "batch_runs", selected_folder)

    metadata = load_metadata(folder_path)
    if metadata is not None:
        fig, ax = plt.subplots()
        for run_id in get_batch_run_keys(folder_path):
            parameters = metadata[run_id]['simulation_parameters']
            variable_parameters = {}
            # Get variable parameters from list of all parameters
            for key, value in parameters.items():
                if key in selected_folder:
                    variable_parameters[key] = value

            agent_out, model_out = load_run_data(run_id, folder_path)
            if model_out is not None:
                ax.plot(model_out['time_step'], model_out['wage'], label=', '.join([f"{key}: {value}" for key, value in variable_parameters.items()]))

        ax.set_xlabel('Step')
        ax.set_ylabel('Wage')
        ax.set_title('Wage vs Step')
        ax.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)

def load_run_data(run_id, folder_path):
    agent_file = os.path.join(folder_path, f"{run_id}_agent.csv")
    model_file = os.path.join(folder_path, f"{run_id}_model.csv")

    if os.path.exists(agent_file) and os.path.exists(model_file):
        agent_out = pd.read_csv(agent_file)
        model_out = pd.read_csv(model_file)
        return agent_out, model_out
    else:
        return None, None

def load_metadata(folder_path, file_path = "run_metadata.yaml"):
    metadata_file = os.path.join(folder_path, file_path)

    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as file:
            metadata = yaml.safe_load(file)
        return metadata
    else:
        st.warning("Metadata file not found.")
        return None

def get_run_ids(folder_path):
    file_names = os.listdir(folder_path)
    run_ids    = set()

    for file_name in file_names:
        if file_name.endswith("_agent.csv"):
            run_id = file_name.replace("_agent.csv", "")
            run_ids.add(run_id)

    return list(run_ids)

def get_batch_run_folders():
    output_data_folder = "output_data"
    runs_folder = "batch_runs"
    batch_run_folders = os.listdir(os.path.join(output_data_folder, runs_folder))
    batch_run_folders = [folder for folder in batch_run_folders if folder != ".DS_Store"]
    return batch_run_folders

def get_batch_run_keys(folder_path):
    file_names = os.listdir(folder_path)
    keys = []

    for file_name in file_names:
        if file_name.endswith("_model.csv"):
            key = file_name.replace("_model.csv", "")
            keys.append(key)

    return keys

def main():
    num_steps  = st.sidebar.slider("Number of Steps", min_value=1, max_value=100, value=10)

    parameters = {
        'width':  10,
        'height': 10,
        'subsistence_wage': st.sidebar.slider("Subsistence Wage", min_value=30000., max_value=50000., value=40000., step=1000.),
        'working_periods':  st.sidebar.slider("Working Periods", min_value=30, max_value=50, value=40),
        'r_prime':          st.sidebar.slider("R Prime", min_value=0.03, max_value=0.07, value=0.05, step=0.01),
        'init_wage_premium_ratio': st.sidebar.slider("Initial Wage Premium Ratio", min_value=0.1, max_value=1.0, value=0.2, step=0.1),
        'gamma':            st.sidebar.slider("Gamma", min_value=0.01, max_value=0.1, value=0.02, step=0.01),
        'discount_rate':    st.sidebar.slider("Discount Rate", min_value=0.05, max_value=0.1, value=0.07, step=0.01),
        'seed_population':  st.sidebar.slider("Seed Population", min_value=1, max_value=100, value=10),
        'density':          st.sidebar.slider("Density", min_value=100, max_value=500, value=300)
    }

    agent_out, model_out = run_simulation(num_steps, parameters) # num_steps, subsistence_wage, working_periods, savings_rate, r_prime)
    
    st.title("Housing Market Model")
    st.header("Model")
    plot_model_data(model_out)

    # Plot heat map for people data
    st.header("People")
    df = agent_out.query("agent_type == 'Person'")
    df = df.dropna(axis=1, how='all').reset_index(drop=True)
    df = df.reset_index(drop=True)

    # Get the list of available variables in the DataFrame
    available_variables = [col for col in df.columns if col not in ['time_step', 'agent_class', 'agent_type', 'id', 'x', 'y']]

    # Create a dropdown menu to select the variable
    selected_variable = st.selectbox("Select variable to plot", available_variables, index=available_variables.index('is_working'))

    # Plot the selected variable on the heatmap
    plot_agent_heatmap(df, selected_variable)

    # Plot heat map for land data
    st.header("Land")
    df = agent_out.query("agent_type == 'Land'")
    df = df.dropna(axis=1, how='all').reset_index(drop=True)
    df = df.reset_index(drop=True)

    # Get the list of available variables in the DataFrame
    available_variables = [col for col in df.columns if col not in ['time_step', 'agent_class', 'agent_type', 'id', 'x', 'y']]

    # Create a dropdown menu to select the variable
    selected_variable = st.selectbox("Select variable to plot", available_variables, index=available_variables.index('warranted_price'))

    # Plot the selected variable on the heatmap
    plot_agent_heatmap(df, selected_variable)

    st.markdown("---")

    st.title("View Batch Run Output")

    # Button to run batch_run.py
    if st.button("Run batch_run.py"):
        # Execute batch_run.py using subprocess
        subprocess.run(["python", "batch_run.py"])

    plot_batch_run_data()

if __name__ == "__main__":
    main()