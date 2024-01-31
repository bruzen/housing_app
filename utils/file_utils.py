import os
import json
import datetime
import random
import string
import subprocess

output_folder = "output_data"

def record_metadata(filepath, run_id=None, num_steps=None, params=None, timestamp=None, batch_parameters=None, variable_parameters=None):
    if variable_parameters:
        metadata = {
            'timestamp':           timestamp, 
            'git_version':         get_git_commit_hash(),
            'batch_parameters':    batch_parameters,
            'variable_parameters': variable_parameters,
            # 'experiment_name':     name,
            # 'fixed_parameters':    fixed_parameters,
            # 'simulation_parameters': model_parameters
        }
    else:
        metadata = {
            'run_id':                run_id,
            'git_version':           get_git_commit_hash(),
            'num_steps':             num_steps, 
            'simulation_parameters': params
            # 'model_description':     self.model_description,
        }
    with open(filepath, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    return metadata

    """Append metadata for each experiment to a metadata file."""

    # # Check if the file exists
    # file_exists = os.path.isfile(metadata_file_path)

    # # If the file exists, load the existing metadata; otherwise, create an empty dictionary
    # if file_exists:
    #     with open(metadata_file_path, 'r') as file:
    #         existing_metadata = yaml.safe_load(file)
    # else:
    #     existing_metadata = {}

    # # Append the metadata for the current experiment to the existing metadata dictionary
    # existing_metadata[run_id] = metadata

    # # Write the updated metadata back to the file
    # with open(metadata_file_path, 'w') as file:
    #     yaml.safe_dump(existing_metadata, file)

def get_run_id(timestamp): # , model_name=None , model_version=None):
    unique_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
    return f"{timestamp}-{unique_id}"
    # Adapt the model name to lowercase and replace spaces with underscores
    # formatted_model_name = model_name.lower().replace(" ", "_")
    # return f"{formatted_model_name}_{timestamp}_{unique_id}_v{model_version.replace('.', '_')}"    

def get_subfolder(folder_name = output_folder, subfolder_name = "data"):
    # Create the subfolder path
    subfolder = os.path.join(folder_name, subfolder_name)
    
    # Create the subfolder if it doesn't exist
    os.makedirs(subfolder, exist_ok=True)
    
    return subfolder

def get_metadata_subfolder():
    return get_subfolder(folder_name = output_folder, subfolder_name = "metadata")

def get_log_subfolder():
    return get_subfolder(folder_name = output_folder, subfolder_name = "logs")

def get_figures_subfolder():
    return get_subfolder(folder_name = output_folder, subfolder_name = "figures")

def get_data_subfolder():
    return get_subfolder(folder_name = output_folder, subfolder_name = "data")

def get_metadata_filepath(file_name):
    subfolder = get_metadata_subfolder()
    return os.path.join(subfolder, file_name)

def get_log_filepath(file_name):
    subfolder = get_log_subfolder()
    return os.path.join(subfolder, file_name)

def get_figures_filepath(file_name):
    subfolder = get_figures_subfolder()
    return os.path.join(subfolder, file_name)

def get_data_filepath(file_name):
    subfolder = get_data_subfolder()
    return os.path.join(subfolder, file_name)

def get_filepath(folder_name=None, subfolder_name=None, file_name=None):
    # Create the subfolder path
    subfolder = os.path.join(folder_name, subfolder_name) if folder_name and subfolder_name else folder_name or subfolder_name
    
    # Create the subfolder if it doesn't exist
    if subfolder:
        os.makedirs(subfolder, exist_ok=True)

    filepath = os.path.join(subfolder, file_name) if subfolder and file_name else subfolder or file_name

    return filepath

def get_git_commit_hash():
    try:
        # Run 'git rev-parse --short HEAD' to get the short commit hash
        result = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
        return result.decode('utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def get_git_details():
    try:
        # Run the git log command to get details of the latest commit, including the commit message and date
        result = subprocess.run(['git', 'log', '-1', '--pretty=format:%cd%n%h%n%an%n%s', '--date=iso'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check if the command was successful
        if result.returncode == 0:
            # The output contains date, commit hash, author, and commit message
            commit_details = result.stdout.strip().split('\n')
            commit_message = commit_details[3]
            
            # Extract as many full words as possible within 30 characters
            words = commit_message.split()
            truncated_message = ''
            char_count = 0
            for word in words:
                if char_count + len(word) <= 30:
                    truncated_message += word + ' '
                    char_count += len(word) + 1  # Account for the space between words
                else:
                    break
            
            formatted_date = commit_details[0][:16]  # Extract the date in 'YYYY-MM-DD HH:mm' format
            return f"{truncated_message.strip()}.. {formatted_date}"
        else:
            # If there was an error, print the error message
            print(f"Error: {result.stderr.strip()}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def generate_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")