import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set up the grid
GRID_SIZE = 20
EMPTY = 0
AGENT_A = 1
AGENT_B = 2

# Initialize the grid with random agent distribution
@st.cache(allow_output_mutation=True)
def initialize_grid():
    return np.random.choice([AGENT_A, AGENT_B], size=(GRID_SIZE, GRID_SIZE), p=[0.5, 0.5])

grid = initialize_grid()
empty_cells = np.where(grid == EMPTY)

# Define the segregation threshold
seg_threshold = st.slider("Segregation Threshold", 0.0, 1.0, 0.3, 0.05)

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
def simulate_step(grid):
    new_grid = grid.copy()
    empty_cells = np.where(grid == EMPTY)

    for i, j in zip(*empty_cells):
        similarity = calculate_similarity(grid, i, j)

        if similarity < seg_threshold:
            agent_type = grid[i, j]

            # Find a random empty cell to move to
            random_index = np.random.randint(len(empty_cells[0]))
            ni, nj = empty_cells[0][random_index], empty_cells[1][random_index]

            new_grid[i, j] = EMPTY
            new_grid[ni, nj] = agent_type

            # Update the empty cells list
            empty_cells[0][random_index] = i
            empty_cells[1][random_index] = j

    return new_grid

# Set up the Streamlit app
st.set_page_config(layout='wide')  # Optional: Set the layout to wide
st.title("Schelling's Model of Segregation")

# Design the user interface
num_steps = st.slider("Number of Steps", min_value=1, max_value=100, value=10)
play_button = st.button("Play")
step_number = st.empty()

# Run the simulation
fig, ax = plt.subplots()
current_step = 0
grid_history = []

for step in range(num_steps):
    if play_button:
        grid_history.append(grid.copy())
        grid = simulate_step(grid)

    ax.imshow(grid, cmap='bwr', vmin=0, vmax=2)
    ax.set_title(f"Step: {step}")
    ax.axis('off')

    st.pyplot(fig)

    if play_button:
        current_step += 1
        step_number.text(f"Current Step: {current_step}")

# Display previous steps
if len(grid_history) > 0:
