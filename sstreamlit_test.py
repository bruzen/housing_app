import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def get_batch_run_folders():
    output_data_folder = "output_data"
    runs_folder = "batch_runs"
    batch_run_folders = os.listdir(os.path.join(output_data_folder, runs_folder))
    return batch_run_folders

def get_batch_run_keys(folder_path):
    file_names = os.listdir(folder_path)
    keys = []

    for file_name in file_names:
        if file_name.endswith("_model.csv"):
            key = file_name.replace("_model.csv", "")
            keys.append(key)

    return keys

def load_data(run_id, folder_path):
    model_file = os.path.join(folder_path, f"{run_id}_model.csv")

    if os.path.exists(model_file):
        model_data = pd.read_csv(model_file)
        return model_data
    else:
        st.error(f"Data file not found for run ID: {run_id}")
        return None

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

    run_ids = get_batch_run_keys(folder_path)
    selected_run_id = st.selectbox("Select Run ID", run_ids)

    data = load_data(selected_run_id, folder_path)
    if data is not None:
        st.subheader("Data")
        st.dataframe(data)

        st.subheader("Plot")
        plot_data(data)

if __name__ == "__main__":
    main()
