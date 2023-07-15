import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml

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

def load_metadata(folder_path):
    metadata_file = os.path.join(folder_path, "run_metadata.yaml")

    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as file:
            metadata = yaml.safe_load(file)
        return metadata
    else:
        st.warning("Metadata file not found.")
        return None

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

        metadata = load_metadata(folder_path)
        if metadata is not None:
            parameters = metadata[selected_run_id]['simulation_parameters']
            st.subheader("Metadata")
            st.write(parameters)

        st.subheader("Plot")
        plot_data(data)

if __name__ == "__main__":
    main()
