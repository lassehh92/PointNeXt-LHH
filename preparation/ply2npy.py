import os
import numpy as np
from pyntcloud import PyntCloud

input_path="/home/lasse/data/Kalundborg_data/validated/Area_2/processed/" # input path to dir containing the plyfiles
output_path="/home/lasse/data/Kalundborg_data/validated/Area_2/processed/npy-files/"  # output path to dir (will be created if not already) where the npy files will end up

# make sure output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# list ply files in the input directory
ply_files = [f for f in os.listdir(input_path) if f.endswith('.ply') and not f.startswith('.')] 

for ply_file in ply_files:
    print (f"{ply_file} is processeing ...")
    cloud = PyntCloud.from_file(os.path.join(input_path, ply_file))
    # print(cloud.points.head())

    # Round x, y, and z columns to 3 decimal places
    cloud.points[['x', 'y', 'z']] = cloud.points[['x', 'y', 'z']].round(3)

    # Remove the scalar fields after color values expect the last field
    cloud.points = cloud.points.iloc[:, [*range(6), -1]]

    # print(cloud.points.head())

    data = np.asarray(cloud.points)
    np.save(os.path.join(output_path,ply_file.replace('ply','npy')), data)


