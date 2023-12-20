import os
import argparse
import re

# Initialize the parser
parser = argparse.ArgumentParser(description="Rename files in a specified directory to ensure continuous Site numbering for a given Area number.")

# Adding arguments
parser.add_argument("dir_path", type=str, help="Directory path where the files are located")
parser.add_argument("area_number", type=int, help="Area number to check and correct Site numbering")

# Parse the arguments
args = parser.parse_args()

folder_path = args.dir_path
area_number = args.area_number

# Function to get site number from filename
def get_site_number(filename):
    match = re.search(r'Site_(\d+)', filename)
    return int(match.group(1)) if match else None

# Collecting site numbers and filenames for the specified area
site_numbers = {}
for filename in os.listdir(folder_path):
    if f"Area_{area_number}" in filename:
        site_num = get_site_number(filename)
        if site_num is not None:
            site_numbers[site_num] = filename

sorted_sites = sorted(site_numbers.keys())

# Rename files to ensure continuous numbering
for i, site_num in enumerate(sorted_sites, start=1):
    if site_num != i:
        old_filename = site_numbers[site_num]
        new_filename = re.sub(r'Site_\d+', f'Site_{i}', old_filename)
        os.rename(os.path.join(folder_path, old_filename), os.path.join(folder_path, new_filename))
        print(f"Renamed '{old_filename}' to '{new_filename}'")

        # Update the dictionary for subsequent renames
        site_numbers[i] = new_filename
        del site_numbers[site_num]

print(f"Renaming completed for Area_{area_number}")
