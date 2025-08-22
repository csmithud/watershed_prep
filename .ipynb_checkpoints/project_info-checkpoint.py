import pathlib as pl
import os
import re
#set project variables
Project_Area = 'nebraska_regression_stantec'
sr = '26852' #set to None if you want to use the DEM's original projection
res = '10m' #DEM resolution, options are '1m', '3m', '10m', '30m', 'OPR'
buffer = 2000 #buffer around the project area in native units
#buffer = 0 #set to 0 if you do not want to buffer the project area
Location = Project_Area+'_'+sr #initiate variable here
Mapset = 'PERMANENT' #always set to this

## Set variables analysis
data_scale = 'HUC8' #other options HUC12, HUC10, HUC8 OR the field name of the source data to be split <-- case sensitive
analysis_scale = 'HUC10'
aoi = '1012010501' #     # Value within the data_scale field used for data selection
if data_scale.find('HUC') >= 0:
    huc_level = re.findall("[0-9]+",data_scale)[0]
    if huc_level == str(len(aoi)):
        select_data = aoi
    else:
        select_data = aoi[:int(huc_level)]
to_headwaters = False #False if only interested in local huc area

#dem info
dem_base_name = 'state_dem' #for saving in grass
aligned = False
carved = True

# provide shapefile with polygons or point to wbd file
basins = pl.Path(os.getcwd()).parent/'data'/'Vectors'/Project_Area/'WBDHU10.shp' 

#if you'd like to clean up after yourself, set this to True
auto_delete = False

#standard folder structure parameters
data_dir = pl.Path(os.getcwd()).parent/'data'
vector_dir = data_dir/'Vectors'/Project_Area
raster_dir = data_dir/'Rasters'/Project_Area
if not os.path.exists(vector_dir):
    os.makedirs(vector_dir)
if not os.path.exists(raster_dir):
    os.makedirs(raster_dir)
    
#prep area of interest
aoi_path = gpd.read_file(basins).loc[temp[analysis_scale.lower()] == aoi].to_file(vector_dir/'aoi.geojson')
