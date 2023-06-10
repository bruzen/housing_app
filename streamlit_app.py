import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set up the grid
GRID_SIZE = 20
EMPTY = 0
AGENT_A = 1
AGENT_B = 2

# Define the segregation threshold
SEGREGATION_THRESHOLD = 0.3

# Initialize the grid with random agent distribution
grid = np.random.choice([AGENT_A, AGENT_B], size=(GRID_SIZE, GRID_SIZE), p=[0.45, 0.45])
empty_cells = np.where(grid == EMPTY)

# Calculate the percentage of similar neighbors
def calculate_similarity(grid, i, j):
    agent_type = grid[i, j]
    total_neighbors = 0
    similar_neighbors = 0

    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue

            ni = i + di
            nj = j + dj

            if ni < 0 or ni >= GRID_SIZE or nj < 0 or nj >= GRID_SIZE:
                continue

            total_neighbors += 1

            if grid[ni, nj] == agent_type:
                similar_neighbors += 1

    if total_neighbors == 0:
        return 0.0

    return similar_neighbors / total_neighbors

# Perform one step of the simulation
def simulate_step():
    global grid

    for i, j in zip(*empty_cells):
        similarity = calculate_similarity(grid, i, j)

        if similarity < SEGREGATION_THRESHOLD:
            agent_type = grid[i, j]

            # Find a random empty cell to move to
            random_index = np.random.randint(len(empty_cells[0]))
            ni, nj = empty_cells[0][random_index], empty_cells[1][random_index]

            grid[i, j] = EMPTY
            grid[ni, nj] = agent_type

            # Update the empty cells list
            empty_cells[0][random_index] = i
            empty_cells[1][random_index] = j

# Set up the Streamlit app
st.title("Schelling's Model of Segregation")

# Design the user interface
num_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=10)

# Run the simulation
fig, ax = plt.subplots()

for step in range(num_steps):
    ax.imshow(grid, cmap='bwr', vmin=0, vmax=2)
    ax.set_title(f"Step: {step}")
    ax.axis('off')

    st.pyplot(fig)

    simulate_step()

# Run the Streamlit app
if __name__ == '__main__':
    st.set_page_config(layout='wide')  # Optional: Set the layout to wide
    st.sidebar.markdown("Adjust the number of steps using the slider.")
    simulate_step()  # Perform an initial simulation step


# import streamlit as st
# import pandas as pd
# import numpy as np
# import pydeck as pdk
# import matplotlib.pyplot as plt


# chart_data = pd.DataFrame(
#    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
#    columns=['lat', 'lon'])

# st.pydeck_chart(pdk.Deck(
#     map_style=None,
#     initial_view_state=pdk.ViewState(
#         latitude=37.76,
#         longitude=-122.4,
#         zoom=11,
#         pitch=50,
#     ),
#     layers=[
#         pdk.Layer(
#            'HexagonLayer',
#            data=chart_data,
#            get_position='[lon, lat]',
#            radius=200,
#            elevation_scale=4,
#            elevation_range=[0, 1000],
#            pickable=True,
#            extruded=True,
#         ),
#         pdk.Layer(
#             'ScatterplotLayer',
#             data=chart_data,
#             get_position='[lon, lat]',
#             get_color='[200, 30, 0, 160]',
#             get_radius=200,
#         ),
#     ],
# ))

# arr = np.random.normal(1, 1, size=100)
# fig, ax = plt.subplots()
# ax.hist(arr, bins=20)

# st.pyplot(fig)
