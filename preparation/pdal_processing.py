import os
import pdal 
import json


input_path="/home/lasse/data/Kalundborg_data/validated/Area_2/" # input path to dir containing the point cloud files
output_path="/home/lasse/data/Kalundborg_data/validated/Area_2/processed/" # output path to dir (will be created if not already) where the processed point cloud files will end up

# make sure output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

# list ply files in input the input directory
pc_files = [f for f in os.listdir(input_path) if f.endswith('.ply') or f.endswith('.las') and not f.startswith('.')] # added startswith condition to aviod inclussion of ._files which tricks pdal error

heating_area_1 = "1  0  0  -633450.00  0  1  0  -6170850.00  0  0  1  0.00  0  0  0  1"    

### PDAL Pipelines ###

### LAS CLASSIFICATIONS ###
# 0 = Never classified
# 1 = Unassigned
# 2 = Ground
# 3 = Low Vegetation
# 4 = Medium Vegetation
# 5 = High Vegetation
# 6 = Building
# 7 = Low Point
# 8 = Reserved
# 9 = Water

def pdal_pipeline_prepare_to_PointNeXt_LAS2PLY(filename):
    pipeline_dict = [
        {
            "type":"readers.las",
            "filename":input_path+filename
        },
        {
            "type":"filters.voxelcenternearestneighbor",
            "cell":0.003
        },
        {
            "type": "filters.ferry",
            "dimensions": "=>Classification"
        },
        {
            "type":"filters.assign",
            "assignment":"Classification[:]=0"
        },
        {
            "type":"filters.outlier",
            "method":"statistical",
            "mean_k":60,
            "multiplier":6.0,
            "class":7
        },
        {
            "type":"filters.groupby",
            "dimension":"Classification"
        },
        {
            "type":"writers.ply",
            "storage_mode":"big endian",
            "filename":os.path.join(output_path, filename.removesuffix(".las")+'_#.ply')
        }
    ]
    return pipeline_dict
 
def pdal_pipeline_prepare_to_PointView_PLY2LAS(filename):
    pipeline_dict = [
        {
            "type":"readers.ply",
            "filename":input_path+filename,
        },
        {
            "type": "filters.ferry",
            "dimensions": "=>Classification"
        },
        {
            "type":"filters.ferry",
            "dimensions":"label => Classification"
        },
        {
            "assignment": "Classification[2:2]=9",
            "type": "filters.assign"
        },
        {
            "assignment": "Classification[1:1]=6",
            "type": "filters.assign"
        },
        {
            "assignment": "Classification[0:0]=2",
            "type": "filters.assign"
        },
        {
            "type":"writers.las",
            "scale_x":"0.0000001",
            "scale_y":"0.0000001",
            "scale_z":"0.0000001",
            "offset_x":"auto",
            "offset_y":"auto",
            "offset_z":"auto",
            "filename":os.path.join(output_path,filename.replace('ply','las'))
        }
    ]
    return pipeline_dict

def pdal_pipeline_prepare_opentrench3d(filename, transform):
    pipeline_dict = [
        {
            "type":"readers.ply",
            "filename":input_path+filename
        },
        {
            "type":"filters.transformation",
            "matrix":transform
        },
        {
            "type":"filters.voxelcenternearestneighbor",
            "cell":0.004
        },
        {
            "type": "filters.ferry",
            "dimensions": "=>Classification"
        },
        {
            "type":"filters.ferry",
            "dimensions":"scalar_Classification => Classification"
        },
        {
            "type":"filters.outlier",
            "method":"statistical",
            "mean_k":60,
            "multiplier":2.2,
            "class":7
        },
        {
            "type":"filters.range",
            "limits":"Classification![7:7]"
        },
        {
            "type":"writers.ply",
            "storage_mode":"big endian",
            "filename":output_path+filename
        }
    ]
    return pipeline_dict

def pdal_pipeline_Heating_Area_1(filename):
    pipeline_dict = [
        {
            "type":"readers.ply",
            "filename":input_path+filename
        },
        {
            "type":"filters.transformation",
            "matrix":"1  0  0  -720030.00  0  1  0  -6183880.00  0  0  1  0.00  0  0  0  1"
        },
        {
            "type":"filters.voxelcenternearestneighbor",
            "cell":0.003
        },
        {
            "type":"writers.ply",
            "storage_mode":"big endian",
            "filename":output_path+filename
        }
    ]
    return pipeline_dict


### Loop point cloud files and execute PDAL Pipeline ###

for pc_file in pc_files:
    print(f"{pc_file} is processing ...")
    pipeline_json = json.dumps(pdal_pipeline_prepare_opentrench3d(pc_file, heating_area_1))
    pipeline = pdal.Pipeline(pipeline_json)
    count = pipeline.execute()
    arrays = pipeline.arrays
    metadata = pipeline.metadata
    log = pipeline.log

