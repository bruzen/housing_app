import os
import yaml
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt


import plotly.graph_objects as go
from plotly.subplots import make_subplots

from model.model import City

@st.cache_data()
def run_model(parameters, num_steps):
    city = City(parameters)
    for t in range(num_steps):
        city.step()

    # Get output data
    agent_out = city.datacollector.get_agent_vars_dataframe()
    model_out = city.datacollector.get_model_vars_dataframe()
    return agent_out, model_out

def plot_output(agent_out, model_out):
    workers = np.array(model_out['workers'])
    wage = np.array(model_out['wage'])
    city_extent = np.array(model_out['city_extent'])
    time = np.arange(len(workers))

    land_out = agent_out.query("agent_type == 'Land'")


    # land_out.to_csv('land_out.csv', index=True)


    # Prepare the data for visualization
    df = land_out.reset_index()

    # Create a list of figures for each step
    figs = []
    for step in df['Step'].unique():
        temp_df = df[df['Step'] == step]
        fig = go.Figure(data=go.Heatmap(
            z=temp_df['warranted_price'],
            x=temp_df['x'],
            y=temp_df['y'],
            hovertext=temp_df['net_rent'],
            colorbar=dict(title='Warranted Price', titleside='right')
        ))
        fig.update_layout(title=f'Step: {step}', xaxis_nticks=20, yaxis_nticks=20)
        figs.append(fig)

    # Use subplot to add a slider through each step
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
                    colorbar=trace.colorbar,
                    visible=(i==1)  # only the first trace is visible
                )
            )

    # Create frames for each step
    final_fig.frames = [go.Frame(data=[figs[i].data[0]], name=str(i)) for i in range(len(figs))]

    # Create a slider to navigate through each step
    steps = [dict(label=str(i), method="animate", args=[[str(i)], dict(frame=dict(duration=300, redraw=True))]) for i in range(len(figs))]

    sliders = [dict(active=0, pad={"t": 50}, steps=steps)]

    final_fig.update_layout(height=600, width=800, title_text="Warranted Price Heatmap Over Steps", sliders=sliders)

    final_fig.show()



    # # Create the heatmap using Seaborn
    # fig = sns.heatmap(agent_out.pivot('y', 'x', 'warranted_price'), annot=True, fmt=".2f", cmap="YlGnBu")

    # # Display the heatmap using Streamlit
    # st.pyplot(fig.figure)

    # # Has slider but maps to a grid
    # # Create a range of time steps from the land_out DataFrame
    # time_steps = land_out.index.get_level_values('Step').unique()

    # # Set Seaborn color palette and style
    # sns.set_palette("YlGnBu", n_colors=10)
    # sns.set_style("white")

    # # Create an empty list to store the heatmap charts
    # heatmap_charts = []

    # # Iterate over each time step
    # for time_step in time_steps:
    #     # Filter land_out for the current time step
    #     data = land_out.xs(time_step, level='Step')
        
    #     # Generate x and y coordinates for the grid
    #     grid_size = int(np.sqrt(len(data))) + 1
    #     x = np.arange(grid_size)
    #     y = np.arange(grid_size)
        
    #     # Create the meshgrid
    #     X, Y = np.meshgrid(x, y)
        
    #     # Prepare the data for the heatmap
    #     df = pd.DataFrame({'x': X.flatten()[:len(data)], 'y': Y.flatten()[:len(data)], 'value': data['warranted_price']})
        
    #     # Create the heatmap using Altair and Vega-Lite
    #     heatmap = alt.Chart(df).mark_rect().encode(
    #         x=alt.X('x:O', axis=alt.Axis(format=".0f")),
    #         y=alt.Y('y:O', axis=alt.Axis(format=".0f")),
    #         color='value:Q',
    #         tooltip=['x:O', 'y:O', 'value:Q']
    #     ).properties(
    #         width=500,
    #         height=500,
    #         title=f"Time Step: {time_step}"
    #     )
        
    #     # Add the heatmap chart to the list
    #     heatmap_charts.append(heatmap)

    # # Create a slider to adjust the time step
    # time_step = st.slider('Time Step', min_value=min(time_steps), max_value=max(time_steps))

    # # Select the corresponding heatmap chart based on the selected time step
    # selected_heatmap = heatmap_charts[time_steps.get_loc(time_step)]

    # # Display the selected heatmap using Streamlit
    # st.altair_chart(selected_heatmap, use_container_width=True)


    # # Shows up
    # # Determine the grid size based on the number of land agents in land_out
    # grid_size = int(np.sqrt(len(land_out))) + 1

    # # Generate x and y coordinates for the grid
    # x = np.arange(grid_size)
    # y = np.arange(grid_size)

    # # Create the meshgrid
    # X, Y = np.meshgrid(x, y)

    # # Prepare the data for the heatmap
    # data = land_out.reset_index()
    # df = pd.DataFrame({'x': X.flatten()[:len(data)], 'y': Y.flatten()[:len(data)], 'value': data['warranted_price']})

    # # Set Seaborn color palette and style
    # sns.set_palette("YlGnBu", n_colors=10)
    # sns.set_style("white")

    # # Create the heatmap using Altair and Vega-Lite
    # heatmap = alt.Chart(df).mark_rect().encode(
    #     x=alt.X('x:O', axis=alt.Axis(format=".0f")),
    #     y=alt.Y('y:O', axis=alt.Axis(format=".0f")),
    #     color='value:Q',
    #     tooltip=['x:O', 'y:O', 'value:Q']
    # ).properties(
    #     width=500,
    #     height=500
    # )

    # # Set the aspect ratio to make it square
    # heatmap = heatmap.configure_view(stroke=None)

    # # Display the heatmap using Streamlit
    # st.altair_chart(heatmap, use_container_width=True)




    # Set up the figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Model Output', fontsize=16)

    # Plot 1: Workers vs Time
    axes[0, 0].plot(time, workers, label='Workers', color='blue')
    axes[0, 0].plot(time, wage, label='Wage', linestyle='--', color='red')
    #  axes[0, 0].plot(time, city_extent, label='City Extent', linestyle='dotted', color='red')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Number of Workers')
    # axes[0, 0].set_title('Workers vs Wage')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True)

  # Plot 2: Wage vs Time
    axes[0, 0].plot(time, workers, label='Workers', color='blue')
    axes[0, 0].plot(time, wage, label='Wage', linestyle='--', color='green')
    axes[0, 0].plot(time, city_extent, label='City Extent', linestyle='dotted', color='red')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Number')
    axes[0, 0].set_title('Workers vs Wage')
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].grid(True)
    
    # Plot 3: Wage vs Workers
    axes[0, 1].plot(workers, wage, color='purple')
    axes[0, 1].set_title('Subplot 2')
    axes[0, 1].set_xlabel('Workers')
    axes[0, 1].set_ylabel('Wage')
    axes[0, 1].grid(True)

    # Plot 4: City Extent vs time
    axes[1, 0].plot(city_extent, time, color='magenta')
    axes[1, 0].set_title('Subplot 3')
    axes[1, 0].set_xlabel('Workers')
    axes[1, 0].set_ylabel('City Extent')
    axes[1, 0].grid(True)

    # Plot 5: Workers vs Wage (subplot 4)
    axes[1, 1].plot(workers, wage, color='brown')
    axes[1, 1].set_title('Subplot 4')
    axes[1, 1].set_xlabel('Workers')
    axes[1, 1].set_ylabel('Wage')
    axes[1, 1].grid(True)

    plt.tight_layout()

    return fig




def display_files():
    # Get the list of run IDs
    folder_path = "output_data"
    run_ids = get_run_ids(folder_path   )

    # Display dropdown to select run ID
    selected_run_id = st.selectbox("Select Run ID", run_ids)

    # Load data based on selected run ID
    run_metadata           = load_metadata(selected_run_id, folder_path)
    agent_out2, model_out2 = load_data(selected_run_id)

    # Display the metadata
    st.subheader("Metadata")
    st.write(run_metadata)

    # TODO what does this do?
    if agent_out2 is not None and model_out2 is not None:
        # Display loaded data
        st.subheader("Agent Data")
        st.dataframe(agent_out2)

        st.subheader("Model Data")  
        st.dataframe(model_out2)


def load_data(run_id):
    agent_file = f"{run_id}_agent.csv"
    model_file = f"{run_id}_model.csv"

    if os.path.exists(agent_file) and os.path.exists(model_file):
        agent_data = pd.read_csv(agent_file)
        model_data = pd.read_csv(model_file)
        return agent_data, model_data
    else:
        # st.error(f"Data files not found for run ID: {run_id}")
        return None, None

def load_metadata(run_id, folder_path):
    metadata_file = folder_path + "/metadata.yaml"

    with open(metadata_file, "r") as file:
        metadata = yaml.safe_load(file)

    run_metadata = metadata.get(run_id)
    return run_metadata

def get_run_ids(folder_path):
    file_names = os.listdir(folder_path)
    run_ids = set()

    for file_name in file_names:
        if file_name.endswith("_agent.csv"):
            run_id = file_name.replace("_agent.csv", "")
            run_ids.add(run_id)

    return list(run_ids)

def main():
    num_steps = st.sidebar.slider("Number of Steps", min_value=1, max_value=100, value=10)

    parameters = {
        'width': 20,
        'height': 3,
        'subsistence_wage': st.sidebar.slider("Subsistence Wage", min_value=30000., max_value=50000., value=40000., step=1000.),
        'working_periods': st.sidebar.slider("Working Periods", min_value=30, max_value=50, value=40),
        'savings_rate': st.sidebar.slider("Savings Rate", min_value=0.1, max_value=0.5, value=0.3, step=0.05),
        'r_prime': st.sidebar.slider("R Prime", min_value=0.03, max_value=0.07, value=0.05, step=0.01)
    }

    agent_out, model_out = run_model(parameters, num_steps) # num_steps, subsistence_wage, working_periods, savings_rate, r_prime)
    
    st.title("Housing Market Model Output")
    fig = plot_output(agent_out, model_out)
    # Display the plots using Streamlit
    st.pyplot(fig)    
    
    st.markdown("---")
    st.header("Explore Existing Data")
    display_files()

if __name__ == "__main__":
    main()