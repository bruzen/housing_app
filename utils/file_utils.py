import os
import yaml
import random
import string
import subprocess

def record_metadata(run_id, metadata, metadata_file_path):
    """Append metadata for each experiment to a metadata file."""

    # Check if the file exists
    file_exists = os.path.isfile(metadata_file_path)

    # If the file exists, load the existing metadata; otherwise, create an empty dictionary
    if file_exists:
        with open(metadata_file_path, 'r') as file:
            existing_metadata = yaml.safe_load(file)
    else:
        existing_metadata = {}

    # Append the metadata for the current experiment to the existing metadata dictionary
    existing_metadata[run_id] = metadata

    # Write the updated metadata back to the file
    with open(metadata_file_path, 'w') as file:
        yaml.safe_dump(existing_metadata, file)

def get_run_id(model_name, timestamp, model_version):
    # Adapt the model name to lowercase and replace spaces with underscores
    formatted_model_name = model_name.lower().replace(" ", "_")
    unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))

    # Create the run_id
    return f"{formatted_model_name}_{timestamp}_{unique_id}_v{model_version.replace('.', '_')}"

def get_subfolder(folder_name = "output_data", subfolder_name = "run_data"):
    # Create the subfolder path
    subfolder = os.path.join(folder_name, subfolder_name)
    
    # Create the subfolder if it doesn't exist
    os.makedirs(subfolder, exist_ok=True)
    
    return subfolder

def get_git_commit_hash():
    try:
        # Run 'git rev-parse HEAD' to get the commit hash
        result = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
        return result.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None
