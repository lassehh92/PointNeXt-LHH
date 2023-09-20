import os
import random
import argparse

# Step 1: Set up argparse to get the directory path
parser = argparse.ArgumentParser(description='Select 15% of .npy files and rename them.')
parser.add_argument('directory_path', help='Path to the directory containing .npy files')
args = parser.parse_args()

# Step 2: Set the directory path from the argument
directory_path = args.directory_path

# Step 3: List all files in the directory
all_files = os.listdir(directory_path)

# Step 4: Filter the list to include only .npy files
npy_files = [f for f in all_files if f.endswith(
    '.npy') and not f.startswith('._')]

# Step 5: Randomly select 15% of the .npy files
selected_files = random.sample(npy_files, int(len(npy_files) * 0.15))

# Step 6: Rename the selected files and keep track of the original names
renamed_files = []
for i, file_name in enumerate(selected_files, start=1):
    new_name = f"Area_6_Site_{i}.npy"
    os.rename(os.path.join(directory_path, file_name), os.path.join(directory_path, new_name))
    renamed_files.append(file_name)

# Step 7: Print a list of the original file names that were renamed
print("The following files were renamed:")
for original_name in renamed_files:
    print(original_name)

# Step 8: Print a message indicating the script has completed
print("Files renamed successfully.")
