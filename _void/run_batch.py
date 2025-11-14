#get parameters
import sys
sys.path.append('..')
from src.grass_functions import*
from project_info import *

#for project
Project_Area = 'nebraska_regression_stantec'
sr = '26852' #set to None if you want to use the DEM's original projection
res = '10m' #meters #DEM resolution, options are '1m', '3m', '10m', '30m', 'OPR'

#dem info turn into class later
dem_preprocessed = False
dem_base_name = 'state_dem' #for saving in grass
aligned = False
carved = True

#initiate class
project = ProjectInformation(Project_Area,sr,res,dem_preprocessed,dem_base_name,aligned,carved)
project.create_output_dirs()
initialize_grass_db(project.Location, project.Mapset, project.sr)
output_geo = []

for num in np.arange(1,76):
    #for run
    data_scale = 'pnt_id'
    analysis_scale = 'pnt_id'
    aoi = num
    geometry = 'point' #
    
    aoi = GrassWatershed(project, data_scale,analysis_scale,aoi,geometry)
    aoi.set_grass_selection()
    aoi.set_dem_name()
    aoi.assign_grass_variables()
    aoi.get_grass_grid_size()
    aoi.get_rough_watershed_data()
    aoi.get_basin_area()
    
    regression_data = RegressionData(project,aoi)
    regression_data.set_aoi()
    regression_data.get_drainage_area()
    regression_data.get_stream_order()
    regression_data.get_basin_shape()
    regression_data.CN_mean = regression_data.get_raster_avg(project.raster_dir/'NE_State_CN_nad.tif','CN_mean')
    regression_data.PRISMyr_mm =regression_data.get_raster_avg(project.raster_dir/'PRISM_yr_NE_Statewide.tif','PRISMyr_mm')
    regression_data.get_main_ch_slope()
    gs.run_command('v.out.ogr', input=  aoi.v_basins ,type = 'area',output = aoi.basins, format = 'GeoJSON')
    
    #initialize
    regression_regions = project.vector_dir/'draft_regression.shp'
    regression_flows = RegressionEquations(project,aoi,regression_data,regression_regions)
    #regionalization
    regression_flows.get_regions()
    #calculate and add flows to geojson vector
    regression_flows.calc_flows()
    regression_flows.add_flows_to_outlet()
    output_geo.append(aoi.outlet)