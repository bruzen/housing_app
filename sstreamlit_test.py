import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def get_batch_run_folders():
    output_data_folder = "output_data"
    runs_folder = "batch_runs"
    batch_run_folders = os.listdir(os.path.join(output_data_folder, runs_folder))
    batch_run_folders = [folder for folder in batch_run_folders if folder != ".DS_Store"]
    return batch_run_folders

def load_data(folder_path):
    model_files = [f for f in os.listdir(folder_path) if f.endswith("_model.csv")]
    data = pd.DataFrame()

    for file_name in model_files:
        file_path = os.path.join(folder_path, file_name)
        run_data = pd.read_csv(file_path)
        data = data.append(run_data, ignore_index=True)

    return data

def plot_data(data):
    plt.plot(data['time_step'], data['wage'])
    plt.xlabel('Step')
    plt.ylabel('Wage')
    plt.title('Wage vs Step')
    st.pyplot()

def main():
    st.title("Batch Run Data Plotter")
    batch_run_folders = get_batch_run_folders()
    selected_folder = st.selectbox("Select Batch Run Folder", batch_run_folders)
    folder_path = os.path.join("output_data", "batch_runs", selected_folder)

    data = load_data(folder_path)
    if not data.empty:
        st.subheader("Plot")
        plot_data(data)

if __name__ == "__main__":
    main()
