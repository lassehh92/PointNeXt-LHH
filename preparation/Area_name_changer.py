import os
import argparse

# Initialize the parser
parser = argparse.ArgumentParser(description="Rename files in a specified directory by changing the area number.")

# Adding arguments
parser.add_argument("dir_path", type=str, help="Directory path where the files are located")
parser.add_argument("original_area_number", type=int, help="Original area number to be replaced in the filenames")
parser.add_argument("new_area_number", type=int, help="New area number to replace in the filenames")

# Parse the arguments
args = parser.parse_args()

folder_path = args.dir_path
original_area_number = args.original_area_number
new_area_number = args.new_area_number

# Renaming process
for filename in os.listdir(folder_path):
    if filename.startswith(f"Area_{original_area_number}"):
        new_filename = filename.replace(f"Area_{original_area_number}", f"Area_{new_area_number}")
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
        print(f"Renamed '{filename}' to '{new_filename}'")
