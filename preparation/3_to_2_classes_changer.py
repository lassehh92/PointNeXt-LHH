import os
import glob
import numpy as np

def process_npy_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all .npy files in the input folder
    npy_files = glob.glob(os.path.join(input_folder, '*.npy'))

    for file in npy_files:
        print(f"Processing file: {file}")

        # Load the data
        data = np.load(file)

        # Step 1: Print the number of unique values in the last column and their counts
        unique_values, counts = np.unique(data[:, -1], return_counts=True)
        print(f"Unique values in the last column: {unique_values}")
        print(f"Counts of the unique values: {counts}")

        # Step 2: Change all "1" values to "0"
        data[data[:, -1] == 1, -1] = 0

        # Step 3: Change all "2" values to "1"
        count_of_2s_before_change = np.count_nonzero(data[:, -1] == 2)
        data[data[:, -1] == 2, -1] = 1

        # Step 4: Check if the column now only consists of 0's and 1's
        only_zeros_and_ones = np.all(np.isin(data[:, -1], [0, 1]))
        print(f"Column now only consists of 0's and 1's: {only_zeros_and_ones}")

        # Step 5: Check if the amount of 1's values is the same as the previous amount of 2's values
        count_of_1s_after_change = np.count_nonzero(data[:, -1] == 1)
        same_amount = count_of_1s_after_change == count_of_2s_before_change
        print(f"Amount of 1's values is the same as the previous amount of 2's values: {same_amount}")

        # Save the modified data to the output folder
        output_file = os.path.join(output_folder, os.path.basename(file))
        np.savetxt(output_file, data)
        print(f"Modified data saved to: {output_file}\n")

# Example usage
input_folder = '/Volumes/LHH-WD-1TB/data/test-2-classes/'  # Replace with your actual input folder path
output_folder = '/Volumes/LHH-WD-1TB/data/test-2-classes/2-classes'  # Replace with your actual output folder path

# Uncomment the following line to run the function
process_npy_files(input_folder, output_folder)

# Note: The paths for input_folder and output_folder need to be set to the actual paths before running the function.

