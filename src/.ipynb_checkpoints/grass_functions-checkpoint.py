import os, sys, copy, shutil, subprocess
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, Markdown, Image
import pathlib as pl
import geopandas as gpd
import rasterio
import pandas as pd
import glob
from osgeo import gdal
import requests
import urllib.parse
import urllib.request
import ssl
import json
import shapely
import io
import re
import csv
from IPython.display import clear_output
from zipfile import ZipFile
from shapely import geometry
from shapely.geometry import shape, Polygon
from collections import defaultdict
import matplotlib.colors as colors
import numpy.ma as ma
import fiona
import math
import psutil
import numpy
import datetime
import requests
import json
import platform

proj_four_dict = {'102704':"+proj=lcc +lat_1=40 +lat_2=43 +lat_0=39.83333333333334 +lon_0=-100 +x_0=500000.0000000002 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs "}

#Define GIS based directory and database (data) storage location
Mapset = 'ML'
wkdir = pl.Path(os.getcwd())
if platform.system() == 'Windows':
    docker = False
else:
    docker = True
if docker:
    pem = '/usr/local/share/stantec-ca.crt'
    # create GRASS GIS runtime environment
    subprocess.check_output(["grass", "--config", "path"]).strip()
    #Define GIS based directory and database (data) storage location
    gisbase = r'/usr/lib/grass78'
    gisdb = "/home/grassdata"
    os.environ['GISBASE'] = gisbase
    #Append python scripts to system path
    grass_pydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(grass_pydir)
    
else:
    pem = '../docker/wincacerts.pem'
    # create GRASS GIS runtime environment
    grass7bin = r'C:\Program Files\GRASS GIS 7.8\grass78.bat'
    startcmd = [grass7bin, "--config", "path"]

    #Define GIS based directory and database (data) storage location
    gisbase = subprocess.check_output(startcmd, text=True).strip()
    gisdb = os.path.join(os.path.expanduser("~"), "Documents\grassdata")

    os.environ['GISBASE'] = gisbase
    os.environ['GRASSBIN'] = r'"C:\Program Files\GRASS GIS 7.8\grass78.bat"'

    #Append python scripts to system path
    grass_pydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(grass_pydir)


# do GRASS GIS imports
from grass.script import*
import grass.script as gs
import grass.script.setup as gsetup
import grass.script.array as garray
from grass.pygrass.modules import Module, ParallelModuleQueue

# for the latest development version use:
#pip install git+https://github.com/zarch/grass-session.git
from grass_session import Session
from grass.script import core as gcore

# simply overwrite existing maps like we overwrite Python variable values
os.environ['GRASS_OVERWRITE'] = '1'
# enable map rendering to in Jupyter Notebook
os.environ['GRASS_FONT'] = 'sans'
# set display modules to render into a file (named map.png by default)
os.environ['GRASS_RENDER_IMMEDIATE'] = 'cairo'
os.environ['GRASS_RENDER_FILE_READ'] = 'TRUE'
os.environ['GRASS_LEGEND_FILE'] = 'legend.txt'

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
logger.root.setLevel(logging.INFO)
logging.basicConfig(format = '[%(asctime)s] [%(levelname)s] [%(module)s] : %(message)s')

from src.data.utils import * 


"""esri compatible functions
"""
def esri_rest_query(service:str,parameters:dict) -> list:
    '''
    takes a url and parameters to 
    query arcGIS REST service for information
    SSL bypassed because of 1 case of verification
    error. This will be updated if an alternative source can be located
    '''
    gcontext = ssl.SSLContext()  # Needed to bypass SSL verification issue
    data = urllib.parse.urlencode(parameters)
    data = data.encode('ascii') # data should be bytes
    req = urllib.request.Request(service, data)
    with urllib.request.urlopen(req,context=gcontext) as response:
        projects = json.loads(response.read())
    return projects

def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

def get_esri_JSON(feat,sr):
    '''Translates geodataframe polygon geometry to esri json format
       for querying esri rest services'''
    coordinates = []
    if feat['type'].find('Polygon')>=0:
        if feat['type'] =='MultiPolygon':
            for poly in feat['coordinates']:
                coordinates.append(poly)
            if len(coordinates[0][0]) == 3:
                z = 'true'
                m = 'false'
            elif len(coordinates[0][0]) == 4:
                z = 'true'
                m = 'true'
            else:
                z = 'false'
                m = 'false'
        elif feat['type'] == 'Polygon':
            coordinates = feat['coordinates']
            if len(coordinates[0]) == 3:
                z = 'true'
                m = 'false'
            elif len(coordinates[0]) == 4:
                z = 'true'
                m = 'true'
            else:
                z = 'false'
                m = 'false'

        esri = {"hasZ" : z,
                "hasM" : m,
                "rings":[coordinates],
                "spatialReference":{'wkid':sr}} #translate geometry variables to esri polygon format

    elif feat['type'].find('Line')>=0:
        coordinates = feat['coordinates']
           
        if len(coordinates[0]) == 3:
            z = 'true'
            m = 'false'
        elif len(coordinates[0]) == 4:
            z = 'true'
            m = 'true'
        else:
            z = 'false'
            m = 'false'

        esri = {"hasZ" : z,
                "hasM" : m,
                "paths":[listit(coordinates)],
                "spatialReference":{'wkid':sr}} #translate geometry variables to esri polygon format
    else:
        return ['Not a geometry feature']

    return esri


"""Functions for Watershed RAS Prep
"""
def get_downstream_huc12(huc,huc4_df):
    if len(huc) < 12:
        huc12 = huc4_df.loc[(huc4_df['huc12'].str.startswith(huc) == True) & (huc4_df['tohuc'].str.startswith(huc) == False)]['huc12'].to_list()[0]
    else:
        huc12 = huc
    return huc12

def lower_pd_cols(df):
    columns = df.columns.to_list()
    lower_dict = {}
    for col in columns:
        lower_dict[col] = col.lower()
    df = df.rename(columns = lower_dict)
    return df

def initialize_grass_db(Location, Mapset, GRASS_GIS_Projection,docker=docker):
    if docker:
        #Start GRASS GIS session
        rcfile = gsetup.init(gisbase, "data/grassdata", "nc_basic_spm_grass7", "user1")

        ## Location
        GrassGIS_Location = os.path.join(gisdb, Location)
        #Location and Mapset
        GrassGIS_LocationMapset = os.path.join(gisdb, Location, Mapset)
        #Setup database for the project. Need to setup the location first, which builds a permanent folder for mapsets

        #Check if the Location is setup. If not setup, create the location for the database
        if os.path.exists(GrassGIS_Location):
            print("Database Location Exists")
        else:
            if GRASS_GIS_Projection in proj_four_dict.keys():
                with Session(gisdb=gisdb, location=Location):
                    gs.run_command('g.proj',location=Location,proj4=f"{proj_four_dict[GRASS_GIS_Projection]}",flags='c')
                    #gs.run_command('g.mapset', mapset=Mapset, location=Location)
                    # run something in PERMANENT mapset:
                    print(gcore.parse_command("g.gisenv", flags="s"))
                
            else:
                with Session(gisdb=gisdb, location=Location, create_opts="EPSG:"+GRASS_GIS_Projection):
                    # run something in PERMANENT mapset:
                    print(gcore.parse_command("g.gisenv", flags="s"))

        if os.path.exists(GrassGIS_LocationMapset):
            print("Database Mapset Exists")
        else:
            with Session(gisdb=gisdb, location=Location, mapset=Mapset,create_opts=""):
                # do something in the test mapset.
                print(gcore.parse_command("g.gisenv", flags="s"))  
        # remove old session
        os.remove(rcfile)
    #Start new session
    rcfile = gsetup.init(gisbase,gisdb, Location, 'PERMANENT')
    # example calls
    print(gs.message('Current GRASS GIS 7 environment:'))
    print (gs.gisenv())



def colorize(array, cmap = 'terrain'):
    normed_data = (array - array.min()) /(array.max() - array.min())
    cm = plt.cm.get_cmap(cmap)
    return cm(normed_data)

def list_directories():
    '''List directories for the current directory
    '''
    li_dirs = [folder for folder in os.listdir(os.getcwd()) if os.path.isdir(os.path.join(os.getcwd(), folder))]
    return li_dirs

def list_files(path: str, file_type: str):
    '''Create a list in the directory given by the path string
    '''
    li_files = [file for file in os.listdir(path) if os.path.splitext(file)[-1] == file_type]
    li_all_files = list()
    for file in li_files:
        li_all_files.append(os.path.join(path, file))
    return li_all_files

def raster_files_dic(path: str):
    '''Create a list in the directory given by the path string. Looks for all tiff and img files. 
    adds them in order of date
    '''
    li_raster_files = defaultdict(list)
    folders = [name for name in os.listdir(path) if os.path.isdir(path/name)]
    
    #remove hidden folders on mac file systems
    for f in folders:
        if f.find('.')>=0:
            folders.remove(f)
            
    folders.sort(key=lambda f: int(re.sub('\D', '', f)), reverse = True)
    for f_dir in folders:
        li_files = [file for file in os.listdir(path/f_dir) if os.path.splitext(file)[-1] in ['.tif','.img']]
        for file in li_files:
            li_raster_files[f_dir].append(os.path.join(path,f_dir, file))
    return li_raster_files

#Bounds based on geometry of geodateframe limits
def bounds_lat_lon(geo_df: gpd.GeoDataFrame, Buffer: 32, Coord_System: int):
    bounds = geo_df['geometry'].bounds
    min_lon, min_lat, max_lon, max_lat = bounds.iloc[0]- Buffer
    pd.DataFrame(bounds.iloc[0]- Buffer).T
    return pd.DataFrame(bounds.iloc[0]- Buffer).T

def rectangular_bounds(geo_df: gpd.GeoDataFrame, Buffer_Factor: float, Coord_System: int, Output_path: str):
    Rectangle=geo_df['geometry'].envelope
    Rectangle = Rectangle.to_crs(epsg = Coord_System)
    RectangleBuffer = Rectangle.scale(Buffer_Factor, Buffer_Factor)
    RectangleBuffer.to_file(Output_path)
    return RectangleBuffer.bounds

def DEM_merged_clipped_reprojected(List_Files, DEM_out:str, dstNodata: int, cutlineDSName:str, cutlineWhere:str, dstSRS:str):
    '''This function takes the DEM rasters, merges the files, clips the files to the
    shapefile extent, and reprojects the results to a coordinate system embedded in the function.
    '''
    MergedRaster=gdal.BuildVRT('Vector_district.vrt', List_Files)
    gdal.Warp(DEM_out, MergedRaster,  cutlineDSName = cutlineDSName, cutlineWhere = cutlineWhere,
                     dstNodata = dstNodata, cropToCutline=True, dstSRS = dstSRS) 
    
def display_data(DEM:str, geo_df: gpd.GeoDataFrame):
    '''This function plots the tiff and boundary of the 
    '''
    ds = gdal.Open(DEM)
    myarray = np.array(ds.GetRasterBand(1).ReadAsArray()).astype(np.float32)
    myarray[myarray == -9999 ] = np.NaN
    myarray_masked = np.ma.masked_invalid(myarray)
    colored_data = colorize(myarray_masked, cmap='terrain')
    coord_system = str(geo_df.crs).split(":")[1]
    df_lat_lon = bounds_lat_lon(geo_df, 0, coord_system)
    min_lon, min_lat, max_lon, max_lat =  df_lat_lon.iloc[0]
    map_center=[geo_df.geometry.centroid.y.iloc[0],geo_df.geometry.centroid.x.iloc[0]]
    m = folium.Map(location = map_center , zoom_start = 11.25, control_scale = True, height ='100%', width= '100%')
    style_function = lambda x :{'fillColor':'None', 'color' : 'black'}
    folium.GeoJson(geo_df.geometry, style_function=style_function).add_to(m)
    folium.raster_layers.ImageOverlay(colored_data, opacity=0.55, \
                                    bounds = [ [max_lat, max_lon],[min_lat, min_lon]]).add_to(m)
    folium.LayerControl().add_to(m)
    return m

def tnm_query(dataset:str,bounds:str,docker) -> list:
    '''
    Uses TNM API service to get product download links 
    returns all relevant data from rest service and must
    be filtered by user to get desired content such as download URLs    
    '''
    import json
    base_url = 'https://tnmaccess.nationalmap.gov/api/v1/products?'
    
    #if docker = true, no need to set ssl to false
    attempts = 0
    while attempts < 50000:
        try:
            parameters = {'datasets':dataset,'bbox':bounds,'outputFormat':'JSON'}
            r = requests.get(base_url,parameters,verify=docker)
            response = json.loads(r.content)
            return response
        except:
            attempts+=1
            print(f'Good url but usgs service error. attempt {attempts}')
            print(r.url)
            bounds = bounds[:bounds.find(',')]+'0'+bounds[bounds.find(','):]
            clear_output(wait=True)

def get_local_nhd(gdf):
    dataset = 'National Hydrography Dataset (NHD) Best Resolution'
    bounds = bounds_lat_lon(gdf,0,4326).iloc[0].to_list()
    bounds_str = [str(a) for a in bounds]
    x1 = numpy.average([bounds[0],bounds[2]])
    y1 = numpy.average([bounds[1],bounds[3]])
    bounds_centroid = str(x1)+','+str(y1)
    #current USGS wbd service
    nhd_service = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?' #base url
    #set parameters for query
    nhd_param = {'outFields':'*','geometry':bounds_centroid,
                 'geometryType':'esriGeometryPoint','inSR':4326,
                 'spatialRel':'esriSpatialRelIntersects',
                 'f':'json','returnGeometry':'false'}
    #get query results
    print('searching for local huc12 basin')
    huc12_out = esri_rest_query(nhd_service,nhd_param)
    if huc12_out['features']:
        local_huc12 = huc12_out['features'][0]['attributes']['huc12']
        print('local huc12 is in', huc12_out['features'][0]['attributes']['states'])
    else:
        local_huc12 = ''
        print('no hucs found. try again later')
    return local_huc12
    
            
def url_is_alive(url:str,docker)-> bool:
    '''
    Checks that a given URL is reachable.
    :param url: A URL
    :rtype: bool
    '''
    

    try:
        response = requests.get(url,verify=docker)
        if response.status_code == 200:
            return True
        else:
            return False
    except urllib.request.HTTPError:
        return False

def national_map_download(s3_url:str,project:str,raster_name:str,output_dir:str) -> list:
    '''
    creates link to raster file on USGS The National Map. Checks link before attempting 
    to download. Prints updates and returns a list of the dems downloaded.
    '''
    ##update in future to get S3 bucket information and only include available files
    
    link = s3_url+project+'/TIFF/'+raster_name
    output_dir_pl = pl.Path(output_dir)
    if output_dir_pl.exists() is False:
        os.makedirs(output_dir_pl)
    dems = []
    print('opening: '+link)
    alive = url_is_alive(link,docker)
    if alive is True:
        r = requests.get(link, allow_redirects=True,verify=False)
        print('saving to: '+str(output_dir_pl/raster_name))
        with open (output_dir_pl/raster_name, 'wb') as tif:
            tif.write(r.content)
        dems.append(raster_name)
    else:
        print('no DEM available. trying next link')
    return dems

def national_map_download_1_arc(dem_name:str,out_dir:pl.Path) -> str:
    '''
    creates link to 1 arc second zip file on USGS The National Map. Checks link before attempting 
    to download. Prints updates and returns a list of the files downloaded.
    '''
    ##update in future to get S3 bucket information and only include available files
    s3_url = 'https://prd-tnm.s3.amazonaws.com'
    link_opt_1 = s3_url+'/StagedProducts/Elevation/1/IMG/'+dem_name
    link_opt_2 = s3_url+'/StagedProducts/Elevation/1/IMG/'+re.findall('n\d{2}w\d{3}',dem_name)[0]+'.zip'
    links = [link_opt_1, link_opt_2]
    if out_dir.exists() is False:
        os.makedirs(out_dir)
    for link in links:
        alive = url_is_alive(link,docker)
        if alive is True:
            print('opening: '+link)
            r = requests.get(link, allow_redirects=True,verify=False)
            memfile = io.BytesIO(r.content)
            with ZipFile(memfile,'r') as openzip:
                print('saving raster to: '+str(out_dir))
                gridfiles = openzip.namelist()
                for local_file in gridfiles:
                    if local_file[-4:] == '.img':
                        f = openzip.open(local_file)
                        content = f.read()
                        f_name = local_file.split('/')[-1]
                        local_file_disk = os.path.join(out_dir,f_name)
                        with open(local_file_disk, 'wb') as asc:
                            asc.write(content)
            return out_dir
        else:
            if link == links[0]: 
                print('attempting second link')
            else:
                print('download attempt failed')
                return 'attempt failed'
                  
def nhd_download(select_data:str,vector_dir:pl.Path) -> str:
    '''
    creates link to nhd zip file on USGS The National Map. Checks link before attempting 
    to download. Prints updates and returns a list of the files downloaded.
    '''
    ##update in future to get S3 bucket information and only include available files
    s3_url = 'https://prd-tnm.s3.amazonaws.com'
    product_name = 'NHDPLUS_H_'+select_data[:4]+'_HU4_GDB.zip'
    link = s3_url+'/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/'+product_name
    gdb = product_name[:-4]+'.gdb'
    gdb_dir = vector_dir/gdb
    if vector_dir.exists() is False:
        os.mkdir(vector_dir)
    else:
        if gdb_dir.exists() is True:
            print('gdb is already downloaded')
            return gdb_dir
    print('opening: '+link)
    alive = url_is_alive(link,docker)
    if alive is True:
        os.mkdir(gdb_dir)
        r = requests.get(link, allow_redirects=True,verify=False)
        memfile = io.BytesIO(r.content)
        with ZipFile(memfile,'r') as openzip:
            print('saving to: '+str(gdb_dir))
            openzip.extractall(path=vector_dir)
        return gdb_dir
    else:
        print('download attempt failed')
        return 'attempt failed'
            
def get_upstream_extent_usgs(selected_huc) -> list:
    '''
    takes a HUC12 and then queries USGS HUC-12 services
    to identify all upstream HUC12s then get the max extent of these
    HUC12 features.
    Use WWF function if not in the US
    '''
    import json
    #intialize lists
    huc12s = []
    #current USGS wbd service
    nhd_service = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?' #base url
    #set parameters for query
    where_c_init = "huc12 = '"+ selected_huc+"'"
    nhd_param = {'outFields':'*','where':where_c_init,
                 'f':'json','returnGeometry':'false'}
    #get query results
    print('searching for upstream huc12 basins')
    local_huc12 = esri_rest_query(nhd_service,nhd_param)
    #start crawling USGS for upstream HUCs
    if not local_huc12:
        return ['service error']
    if local_huc12['features']:
        huc12_upstream_target = local_huc12['features']
        while huc12_upstream_target:
            for huc12_upstream in huc12_upstream_target:
                #add huc12 to ruinning list, search for more upstream basins, remove from target list
                huc12s.append(huc12_upstream['attributes']['huc12'])
                huc12_upstream_target.remove(huc12_upstream)
                where_c = "tohuc = '"+ huc12_upstream['attributes']['huc12']+"'"
                nhd_param = {'outFields':'huc12','where':where_c,'f':'json','returnGeometry':'false'}
                huc12_upstream = esri_rest_query(nhd_service,nhd_param)['features']
                for target in huc12_upstream:
                    if target not in huc12_upstream_target:
                        huc12_upstream_target.append(target)
                
        print('found '+str(len(huc12s))+' upstream huc12s')
        return huc12s
    else:
        print('Not in USA')
        return ['not in USA']

def get_just_upstream_basins_usgs(selected_huc) -> list:
    '''
    takes a HUC12 and then queries USGS HUC-12 services
    to identify all upstream HUC12s then get the max extent of these
    HUC12 features.
    Use WWF function if not in the US
    '''
    import json
    #intialize lists
    huc12s = []
    huc12s.append(selected_huc)
    #current USGS wbd service
    nhd_service = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?' #base url
    #set parameters for query
    where_c = "tohuc = '"+ selected_huc+"'"
    nhd_param = {'outFields':'huc12','where':where_c,'f':'json','returnGeometry':'false'}
    #get query results
    print('searching for just upstream huc12 basins')
    huc12_upstream = esri_rest_query(nhd_service,nhd_param)['features']
    #start crawling USGS for upstream HUCs
    if not huc12_upstream:
        print('service error')
        return ['service error']
    if huc12_upstream['features']:
        huc12_just_upstream_target = huc12_upstream['features']
        for huc12_upstream in huc12_just_upstream_target:
            #add huc12 to ruinning list, search for more upstream basins, remove from target list
            huc12s.append(huc12_upstream['attributes']['huc12'])
            
        print('found '+str(len(huc12s))+' upstream huc12s')
        return huc12s
    else:
        print('Headwaters HUC')
        return huc12s

def get_upstream_extent_wwf(inputs_dir:pl.Path,point:list,sr:str) -> list:
    '''
    takes a point and then queries WWF Hydro basin shapefiles stored on the docker image
    to identify all upstream HUC12s then get the max extent of these
    HUC12 features.
    '''
    #intialize lists
    huc12s = []
    huc12_exts = []
    
    #convert input point to a geopandas df
    df_point = pd.DataFrame(
    {'Name':'Area of Interest','Lat':[point[1]],'Long':[point[0]]})
    gpd_point =  gpd.GeoDataFrame(df_point, geometry=gpd.points_from_xy(df_point.Long, df_point.Lat))
    gpd_point.set_crs(epsg=int(sr),inplace=True)
    
    #save point as shapefile for later use
    gpd_point.to_file(inputs_dir/"aoi.shp")
    
    #read North American and Arctic Hydro Basins
    print('reading in basins')
    na_no_ca_basins = gpd.read_file('zip://'+str(inputs_dir/'hybas_na_lev12_v1c.zip!hybas_na_lev12_v1c.shp'))
    ca_basins = gpd.read_file('zip://'+str(inputs_dir/'hybas_ar_lev12_v1c.zip!hybas_ar_lev12_v1c.shp'))
    na_basins = na_no_ca_basins.append(ca_basins)
    
    #convert to sr
    na_basins = na_basins.to_crs(epsg=int(sr))
    
    #get query results.... Curtis to update using intersects for faster response
    local_huc12 = gpd.sjoin(gpd_point,na_basins,how='inner',op='intersects')['HYBAS_ID'].to_list()
    if not local_huc12:
        print('point coordinates are not within North America')
        return ['point coordinates are not within North America','none']
    #start crawling USGS for upstream HUCs
    print('getting upstream basins')
    huc12_upstream_target = local_huc12
    while huc12_upstream_target:
        for huc12_upstream in huc12_upstream_target:
            #add HUC12 to running list, save boundaries and remove from target list
            huc12s.append(huc12_upstream)
            bounds = na_basins.loc[na_basins['HYBAS_ID'] == huc12_upstream].bounds
            huc12_exts.append(bounds.iloc[0].to_dict())
            huc12_upstream_target.remove(huc12_upstream)

            #get more target basins that flow into this huc12
            mask = na_basins['NEXT_DOWN'] == huc12_upstream
            up_basins_gdf = na_basins.loc[mask]
            huc12s_upstream = up_basins_gdf['HYBAS_ID'].to_list()
            #add targets to target list
            for target in huc12s_upstream:
                if target not in huc12_upstream_target:
                    huc12_upstream_target.append(target)
    print('found '+str(len(huc12s))+' upstream huc12s')
    
    #set initial bounding box
    [xmin,ymin,xmax,ymax] = [huc12_exts[0]['minx'],huc12_exts[0]['miny'],huc12_exts[0]['maxx'],huc12_exts[0]['maxy']]
    
    #expand to maximum size
    for huc_ext in huc12_exts[1:]:
        if huc_ext['minx'] < xmin:
            xmin = huc_ext['minx']
        if huc_ext['miny'] < ymin:
            ymin = huc_ext['miny']
        if huc_ext['maxx'] > xmax:
            xmax = huc_ext['maxx']
        if huc_ext['maxy'] > ymax:
            ymax = huc_ext['maxy']
    
    #return max boundaries
    bound_box = [xmin,ymin,xmax,ymax]
    print('done')
    return bound_box, huc12s

def bound_box_str(gdf:gpd.GeoDataFrame,increment:int):
    features = []
    ext_df = gdf.bounds.to_dict()
    idx = gdf.index.to_list()
    iterate = range(increment)
    for ix in idx:
        startY_big = ext_df['miny'][ix]
        startX_big = ext_df['minx'][ix]
        diffY = (ext_df['maxy'][ix] - startY_big)/increment
        diffX =  (ext_df['maxx'][ix] - startX_big)/increment
        for x,y in [(x,y) for x in iterate for y in iterate]:
            startY = startY_big + diffY*(y)
            startX = startX_big + diffX*(x)
            endY = startY_big + diffY*(y+1)
            endX = startX_big + diffX*(x+1)
            b_box = [str(round(startX,8)),str(round(endY,8)),str(round(endX,8)),str(round(startY,8))]
            features.append(', '.join(b_box))
    return features

def dem_download(dem_links:list,output_dir:str,dem_folder:list,docker=pem) -> list:
    '''
    Downloads links to DEM data. Does not check link before attempting 
    to download. Prints updates and returns a list of the dems downloaded.
    
    Need to add overwrite option
    '''
    dems = []
    total = len(dem_links)
    for link, fol in zip(dem_links,dem_folder):
        number = dem_links.index(link) + 1
        print('Obtaining DEM '+str(number)+ ' of '+str(total))
        output_dir_pl = pl.Path(output_dir)/fol
        if output_dir_pl.exists() is False:
            os.makedirs(output_dir_pl)

        raster_name = link.split('/')[-1]
        if (output_dir_pl/raster_name).exists() is False:
            print('opening: '+link)
            r = requests.get(link, allow_redirects=True,verify=docker)
            print('saving to: '+str(output_dir_pl/raster_name))
            with open (output_dir_pl/raster_name, 'wb') as tif:
                tif.write(r.content)
            dems.append(raster_name)
           
        else:
            print('DEM already downloaded')
        clear_output(wait=True)
    return dems

# def dem_download(dem_links:list,output_dir:str,dem_folder:list,docker=docker) -> list:
#     '''
#     Downloads links to DEM data. Checks link before attempting 
#     to download. Prints updates and returns a list of the dems downloaded.
    
#     Need to add overwrite option
#     '''
#     dems = []
#     total = len(dem_links)
#     for link, fol in zip(dem_links,dem_folder):
#         number = dem_links.index(link) + 1
#         print('Obtaining DEM '+str(number)+ ' of '+str(total))
#         output_dir_pl = pl.Path(output_dir)/fol
#         if output_dir_pl.exists() is False:
#             os.makedirs(output_dir_pl)

#         raster_name = link.split('/')[-1]
#         if (output_dir_pl/raster_name).exists() is False:
#             alive = url_is_alive(link,docker)
#             if alive:
#                 print('opening: '+link)
#                 r = requests.get(link, allow_redirects=True,verify=False)
#                 print('saving to: '+str(output_dir_pl/raster_name))
#                 with open (output_dir_pl/raster_name, 'wb') as tif:
#                     tif.write(r.content)
#                 dems.append(raster_name)
#             else:
#                 #try saving link with rocky web link which sometimes has missing S3 data
#                 rocky_link = link.replace("https://prd-tnm.s3.amazonaws.com/StagedProducts/","https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/")
#                 alive = url_is_alive(rocky_link,False)
#                 if alive:
#                     print('opening: '+rocky_link)
#                     r = requests.get(rocky_link, allow_redirects=True,verify=docker)
#                     print('saving to: '+str(output_dir_pl/raster_name))
#                     with open (output_dir_pl/raster_name, 'wb') as tif:
#                         tif.write(r.content)
#                     dems.append(raster_name)
#                 else:
#                     print('no DEM available. trying next link')
#         else:
#             print('DEM already downloaded')
#         clear_output(wait=True)
#     return dems

def create_tnm_gdf(responses:list):
    wkid = 4269 #NAD83
    epsg = 'EPSG:'+str(wkid)
    df = pd.DataFrame(responses)
    if df.empty:
        print('1m not available')
        return ['1m not available']
    else:
        b = df.apply(lambda row: geometry.box(row.boundingBox['minX'], row.boundingBox['minY'], row.boundingBox['maxX'], row.boundingBox['maxY']), axis=1)
        gdf = gpd.GeoDataFrame(df, geometry=b,crs = epsg)
    return gdf

def tnm_coverage_check(gdf,aoi_coverage_buffer,projection,minimum_coverage = 0.95):
    'minimum % coverage at 95% hard coded'
    if gdf.empty:
        return False
    else:
        aoi_coverage = aoi_coverage_buffer.to_crs(epsg = projection)
        gdf_coverage = gdf.dissolve().to_crs(epsg = projection)
        inter_area = gdf_coverage.intersection(aoi_coverage)
        coverage_perc = inter_area.area.iloc[0]/aoi_coverage.area.iloc[0]
        if coverage_perc >= minimum_coverage:
            coverage = True
        else:
            coverage = False
        return coverage, coverage_perc
        

def tnm_coverage(aoi, desired_resolution='1m',buffer = 2000, minimum_coverage = 0.95):
    projection = '5070' #used for equal area across the US
    #create 2000' buffer
    aoi_coverage = aoi.dissolve().to_crs(epsg = projection)
    aoi_coverage_buffer = aoi_coverage.buffer(buffer).to_crs(epsg = '4269') #nad83
    if desired_resolution == 'opr':
        tiles=20
    else:
        tiles=4
    bounds = bound_box_str(aoi_coverage_buffer,tiles) #get bounding box  representation of geometry 
    DEM_products = {'1m':'Digital Elevation Model (DEM) 1 meter',
                    '3m': 'National Elevation Dataset (NED) 1/9 arc-second','10m':'National Elevation Dataset (NED) 1/3 arc-second', '30m':'National Elevation Dataset (NED) 1 arc-second',
                   'opr': 'Original Product Resolution (OPR) Digital Elevation Model (DEM)'}
    covered = False
    assert desired_resolution in list(DEM_products.keys()), "Desired DEM resolution can only be 1m, 3m, 10m, 30m, or opr" 
    for res in list(DEM_products.keys())[list(DEM_products.keys()).index(desired_resolution):]:
        tnm_responses = []
        dataset = DEM_products[res]
        for huc12_bound in bounds:
            tnm_out = tnm_query(dataset,huc12_bound,docker)
            attmpt = 1
            while 'items' not in tnm_out.keys():
                print('missed api, retrying'+str(attmpt))
                attmpt += 1
                clear_output(wait=True)
                tnm_out = tnm_query(dataset,huc12_bound,docker)
            if len(tnm_out['items']) == 100:
                print('need larger increment used. reset increment')
            tnm_responses += tnm_out['items']
        if tnm_responses:
            tnm_gdf_out = create_tnm_gdf(tnm_responses)
            tnm_gdf_out['dl_link'] = tnm_gdf_out['urls'].apply(lambda x: list(x.values())[0])
            tnm_gdf_clean = tnm_gdf_out.drop_duplicates(subset='dl_link')
            #trial by date
            collection_dates = list(tnm_gdf_clean["publicationDate"].unique())
            collection_dates.sort(reverse=True,key=lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')) 
            #add dates until coverage check is met
            covered, coverage_perc = tnm_coverage_check(tnm_gdf_clean,aoi_coverage_buffer,projection,minimum_coverage)
            if covered is True:
                print('Best available DEM is '+res)
                print('coverage is %s percent' %(round(coverage_perc,3)*100))
                return tnm_gdf_clean, aoi_coverage_buffer
            # below code sunsetted to get all relevant DEMs
            # for i, collection in enumerate(collection_dates):
            #     gdf_select = tnm_gdf_clean.loc[tnm_gdf_clean["publicationDate"].isin(collection_dates[:i+1]) == True]
            #     covered, coverage_perc = tnm_coverage_check(gdf_select,aoi_coverage_buffer,projection,minimum_coverage)
            #     if covered is True:
            #         print('Best available DEM is '+res)
            #         print('coverage is %s percent' %(round(coverage_perc,3)*100))
            #         return gdf_select, aoi_coverage_buffer
        else:
            covered, coverage_perc = [False,0.0]
        if covered is True:
            print('Best available DEM is '+res)
            print('coverage is %s percent' %(round(coverage_perc,3)*100))
            return tnm_gdf_clean, aoi_coverage_buffer
            
        else:
            print(res +' not available. Checking next best resolution')
            if res == 'opr':
                print('No high quality LiDAR coverage for this project area')
                return [None,None]

def project_raster(raster_name, raster_dir, sr):
    """ Re-projects the raster_file in raster_dir to the provided sr and saves as VRT in raster_dir"""
    gdal.Warp(
        os.path.join(raster_dir, raster_name + "_sr" + sr + ".vrt"),
        os.path.join(raster_dir, raster_name),
        dstSRS='EPSG:'+sr, dstNodata=0
    )
    return os.path.join(raster_dir, raster_name + "_sr" + sr + ".vrt")

def dem_vert_unit_check(dem_source,raster_name,vert_unit):
    vert_value = {'feet':float(0.3048),'meter':float(1)}
    meta = gs.parse_command('g.proj', georef= dem_source, flags='g')
    vert_value_m = float(meta['meters'])
    vert_conversion = (vert_value_m/vert_value[vert_unit])
    if vert_conversion != float(1):
        logger.info('converting vertical units to '+vert_unit+'. Raw DEM * '+str(vert_conversion))
        gs.run_command('g.region', raster=raster_name)
        gs.mapcalc("new_rast = {0} * {1}".format(raster_name, vert_conversion))
        gs.run_command('g.rename', raster= "new_rast,{}".format(raster_name))
        gs.run_command('r.support', map= raster_name, units='ft')
    else:
        gs.run_command('r.support', map= raster_name, units='m')
        
    return None

def mask_dem(config):
    dem = config.dem.name
    logger.info("setting mask to buffered huc")
    gs.run_command('r.mask', vector="hucbuff")
    logger.info("clipping dem to buffered huc and setting region")
    reg = gs.parse_command('g.region', raster=dem, flags='pgm', zoom=dem,align=dem)
    logger.info(reg)


    
def dem_tiles_to_gis(gs,dem_dir,Project_Area,vert_unit, patch=False,delete_raw=False,force=False):
    """imports rasters to given grass session in gs"""
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    dem_name = Project_Area + '_dem'
    dem_boundary_v = dem_name+'_boundary'
    if dem_name in GRASS_raster_files:
        print('dem already created, skipping import and patching. Use Force = True to recreate dem')
        return dem_name
    else:
        tile_rasters = []
        dem_b_rasters = [] 
        clean_up_rasters = [] #hold raster list for later cleanup
        li_all_files = raster_files_dic(dem_dir)
        if not li_all_files:
            print('dem already imported and deleted. Must redownload')
            return None
        total_rasters = sorted({x for v in li_all_files.values() for x in v})
        for collection, rasters in li_all_files.items():
            for raster in rasters:
                name = ('%s' % pl.Path(raster).name)
                ext = name[name.find('.'):]
                raster_name = ('%s' % pl.Path(raster).name)[:-4]
                vrt_path = os.path.join(pl.Path(raster).parent, raster_name + ".vrt")
                raster_name_u = f'{raster_name}_{collection}'.replace('-','_')
                if raster_name not in GRASS_raster_files:
                    print("adding raster {}".format(raster_name_u))
                    gs.run_command('r.in.gdal', input=vrt_path, output= raster_name_u)
                    gs.run_command('g.region', raster=raster_name_u) 
                    dem_vert_unit_check(raster,raster_name_u,vert_unit)

                tile_rasters.append(raster_name_u)
                if delete_raw is True:
                    os.unlink(raster)

                clear_output(wait=True)
        if not tile_rasters:
            for dir_path, dir_names, file_names in os.walk(dem_dir):
                tile_rasters += [file.split(ext)[0] for file in file_names]
        if patch is True:
            print("Patching together {} rasters".format(len(tile_rasters)))

            if len(total_rasters) > 1:
                gs.run_command('g.region', raster=tile_rasters)
                gs.run_command('r.patch', input=tile_rasters, output=dem_name)

            else:
                gs.run_command('g.rename', raster=(tile_rasters[0],dem_name))
    #             gs.run_command('g.rename', raster=(dem_b_rasters[0],dem_name))

            ##fill any gaps in the dem
            #gs.run_command('r.fill.gaps',input=DEM,output='tmp', method)

            print('Created: '+ dem_name)
    #         print('Created: '+ dem_name+'_b')

            #cleanup and remove raster tiles
            for tile in tile_rasters:
                gs.run_command('g.remove', type ='raster', name=tile, flags ='f')
    #             gs.run_command('g.remove', type ='raster', name=tile+'_b', flags ='f')
            return dem_name
        else:
            return tile_rasters


def list_existing_grass():
    #List Existing Files: Vectors and Rasters
    layers = {'vector':[],'raster':[]}
    print('Available vector maps:')
    for vect in gs.list_strings(type='vector'):
        print (vect)
        layers['vector'].append(vect)

    print('\nAvailable raster maps:')
    for rast in gs.list_strings(type='raster'):
        print (rast)
        layers['raster'].append(rast)
    return layers

def remove_grass_data(grass_maps:list,to_remove_name:str,map_type:str = 'None'):
    #option to remove by type or name
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    GRASS_vector_files = [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
    if to_remove_name != 'None':
        if to_remove_name == 'all':
            if map_type == 'raster':
                for layer in GRASS_raster_files:
                    gs.run_command('g.remove', type = map_type, name=layer,flags ='f')
            elif map_type == 'vector':
                for layer in GRASS_vector_files:
                    gs.run_command('g.remove', type = map_type, name=layer,flags ='f')
                    
        if type(to_remove_name) == list:
            for layer in to_remove_name:
                gs.run_command('g.remove', type = map_type, name=layer,flags ='f')
        else:
            gs.run_command('g.remove', type = map_type, name=to_remove_name,flags ='f')
    else:
        pass

def nhd_to_grass(git_data_repo, vector_dir,select_data,upstream_list,huc12s,selected_huc12,upstream_huc12s,pour_points,pour_points_upstream,nhd,upstream_nhd,force= True):
    GRASS_vector_files = [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]

    #Import HUC-12 Boundaries into Grass (importing and clipping can take time)
    if huc12s not in GRASS_vector_files or force:
        gs.run_command('v.import', input= vector_dir/('NHDPLUS_H_'+select_data[:4]+'_HU4_GDB.gdb'), layer = 'WBDHU12',  output= huc12s)   

    #Extract HUC-12 Boundary of Interest in Grass  
    if selected_huc12 not in GRASS_vector_files or force:
        gs.run_command('v.extract', input =huc12s, where= "HUC12='%s'" % select_data, output = selected_huc12)  

    #Extract upstream HUC12s
    if upstream_huc12s not in GRASS_vector_files or force:
        if 'service error' not in upstream_list and 'not in USA' not in upstream_list: #need to add in process for downloading upstream HUC4 if needed
            if len(upstream_list) > 1:
                gs.run_command('v.extract', input = huc12s, where= "HUC12 IN %s" % str(tuple(upstream_list)), output = upstream_huc12s)  
            else:
                gs.run_command('v.extract', input = huc12s, where= "HUC12 = '%s'" % (upstream_list[0]), output = upstream_huc12s)  
    
    #set region for import of NHD data
    region = gs.parse_command('g.region',vector = upstream_huc12s, flags='pg')
    Buffer = round(min([int(region['cols']),int(region['rows'])])*0.075)
    #include buffer
    gs.parse_command('g.region',vector = upstream_huc12s, flags='pg', grow=Buffer)

    #Import pour points 
    if pour_points not in GRASS_vector_files or force:
        gs.run_command('v.import', input=  git_data_repo/'pour_points'/'tpp.shp',extent='region', output= pour_points)

    #Extract upstream HUC12 pour points  
    if pour_points_upstream not in GRASS_vector_files or force:
        if len(upstream_list) > 1:
            gs.run_command('v.extract', input =pour_points, where= "HUC_12 IN %s" % str(tuple(upstream_list)), output = pour_points_upstream) 
        else:
            gs.run_command('v.extract', input =pour_points, where= "HUC_12 = '%s'" % (upstream_list[0]), output = pour_points_upstream) 

    #import NHD burn lines
    if nhd not in GRASS_vector_files or force:
        gs.run_command('v.import', input=  vector_dir/('NHDPLUS_H_'+select_data[:4]+'_HU4_GDB.gdb'),extent='region', layer = 'NHDFlowline',  output= nhd)

    #Extract NHD flow lines for upstream HUC12 areas
    if upstream_nhd not in GRASS_vector_files or force:
        gs.run_command('v.buffer',input=upstream_huc12s,output=upstream_huc12s+'_buffer',distance = 900)
        gs.run_command('v.select', ainput = nhd,binput = upstream_huc12s+'_buffer', output = upstream_nhd, operator = 'intersects') 


        
def nhd_to_grass_local(vector_dir,basins,huc12,huc12s,upstream_huc12s,nhd,upstream_nhd,force= True):
    GRASS_vector_files = [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]

    #Import HUC-12 Boundaries into Grass (importing and clipping can take time)
    if huc12s not in GRASS_vector_files or force:
        gs.run_command('v.import', input= vector_dir/('NHDPLUS_H_'+huc12[:4]+'_HU4_GDB.gdb'), layer = 'WBDHU12',  output= huc12s)   

    #Extract upstream HUC12s
    if upstream_huc12s not in GRASS_vector_files or force:
        gs.run_command('v.buffer',input=basins,output=basins+'_int_buffer',distance = -1000)
        gs.run_command('v.select', ainput = huc12s, binput= basins+'_int_buffer', output = upstream_huc12s)  
    
    #import NHD burn lines
    if nhd not in GRASS_vector_files or force:
        gs.run_command('v.import', input=  vector_dir/('NHDPLUS_H_'+huc12[:4]+'_HU4_GDB.gdb'),extent='region', layer = 'NHDFlowline',  output= nhd)

    #Extract NHD flow lines for upstream HUC12 areas
    if upstream_nhd not in GRASS_vector_files or force:
        gs.run_command('v.buffer',input=basins,output=basins+'_buffer',distance = 900)
        gs.run_command('v.select', ainput = nhd,binput = basins+'_buffer', output = upstream_nhd, operator = 'intersects') 
        
        
        
        
def grass_watershed_processing(basin_scale,upstream_list,upstream_huc12s,upstream_nhd,vector_dir,raster_dir,filter_size = '23', carve = False, force=False):
    '''get accumulation and drain direction, create streams, snap outlets delineate
    post processing of watersheds done separately'''
    
    #get list of dems:
    basin_dems = {}
    dem_dict = {'HUC8':8,'HUC10':10,'HUC12':12}
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    for basin in upstream_list:
        dem_name = 'dem_'+basin[:dem_dict[basin_scale]]
        if dem_name in GRASS_raster_files: 
            basin_dems[basin] = dem_name
        else:
            print(dem_name,'not in Grass raster files')

    #get full region with buffer
    #create a 2000 meter buffer around the HUC-12 to account for differences between the high resolution DEM and NED
    gs.run_command('v.buffer',input=upstream_huc12s,output=upstream_huc12s+'_buffer',distance = 2000)
    
    #set the region based on the buffer but align with DEM raster
    gs.run_command('g.region',vector = upstream_huc12s+'_buffer',align=dem_name)
        
    #Get DEM resolution 
    meta_dem = gs.parse_command('r.info',map=dem_name,flags='g')
    DEM_Res = float(meta_dem['nsres'])
    
    #set variables
    Stream_Threshold_Local= 2590000/DEM_Res #1 sq miles   
    basin_list = []
    huc_key = {}
    if float(upstream_list[0]) > float(upstream_list[-1]):
        upstream_list.reverse()
    outlet_keys = outlet_key(basin_scale,upstream_list)
    basin_name = 'v_domains.shp'
    
    #perform grass processing
    for huc12 in upstream_list:  
        #initiate variables
        value = huc12[:dem_dict[basin_scale]] #trim value to desired huc level
        short_value = huc12[-7:dem_dict[basin_scale]] #small enough to retain int raster status
        GRASS_vector_files= [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
        GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
        draindir = 'r_drain_dir_'+huc12
        outlet_r = 'r_outlet_'+huc12
        accum = 'r_accum_'+huc12
        basins = 'basin_'+huc12
        r_stream = 'stream_'+huc12
        v_stream = 'v_stream_'+huc12
        v_outlet = 'v_outlet_'+huc12
        v_basins = 'v_'+basins  
        outlet_name ='v_'+huc12+'_outlet.shp'
        DEM = basin_dems[huc12]
        gtiff_temp = basins+'.tif'
        if os.path.exists(raster_dir/gtiff_temp): 
            print('data already created for',huc12)
            basin_list.append('out_'+basins)
            huc_key[short_value] = value
            continue
        
        #extract huc12 area of interest
        gs.run_command('v.extract', input =upstream_huc12s, where= "HUC12='%s'" % huc12, output = 'tmp_huc')
          
#         calculate the approximate watershed area 
        ws_calc = gs.parse_command('v.to.db', map='tmp_huc', option='area',flags='pc',columns ='area')
        huc_ws_area = float(list(ws_calc)[-1].split('|')[1])
               
        #create a 2000 meter buffer around the HUC-12 to account for differences between the high resolution DEM and NED
        gs.run_command('v.buffer',input='tmp_huc',output='tmp_huc_buffer',distance = 2000)
        
        #create a slightly smaller buffer around the HUC-12 to avoid carving null cells
        gs.run_command('v.buffer',input='tmp_huc',output='tmp_huc_buffer_nhd',distance = 2000*0.95)
        
        #set the region based on the buffer but align with DEM raster
        gs.run_command('g.region',vector = 'tmp_huc_buffer',align=DEM)
        
        #mask outside cells
        gs.run_command('r.mask',vector = 'tmp_huc_buffer')

        #get memory for watershed scripts
        max_mem = get_RAM()
        
        if carve: 
            if filter_size:
                print('Median filtering of mesh for '+huc12)
                if 'wsd_filt_dem_'+huc12+'_b' not in GRASS_raster_files or force is True:
                    filter_odd = (filter_size//DEM_Res) // 2 * 2 +1
                    gs.run_command('r.neighbors', input = DEM,output='filt_dem_'+huc12,method='median',size=filter_odd)
                DEM = 'filt_dem_'+huc12
            else:
                pass
                
            print('Processing stream lines for '+huc12)    
            if 'nhd_'+accum not in GRASS_raster_files or force is True:
                #extract nhd for area of interest
                gs.run_command('v.overlay', ainput = upstream_nhd,binput = 'tmp_huc_buffer_nhd', output = v_stream, operator = 'and',flags='t')
 
            if v_stream not in GRASS_vector_files or force is True:
#                 max_weight_accum =  gs.parse_command('r.info',map = 'weight_%s' %(accum),flags='s')['max']
#                 gs.run_command('r.stream.extract',elevation = 'filt_r_'+DEM,accumulation=accum+'_r',memory = max_mem,threshold = Stream_Threshold_Local,stream_vector=v_stream)
#                 gs.run_command('v.db.renamecolumn',map=v_stream,column='Permanent_Identifier,pid')
#                 gs.run_command('v.db.renamecolumn',map=v_stream,column='WBArea_Permanent_Identifier,wbapid')
#                 gs.run_command('v.db.renamecolumn',map=v_stream,column='VisibilityFilter,vf')
#                 gs.run_command('v.db.renamecolumn',map=v_stream,column='Shape_Length,shp_l')

                gs.run_command('v.out.ogr',input=v_stream,output=vector_dir,type= 'line',format='ESRI_Shapefile')
                
            print('Carving DEM for '+huc12)
            #carving with streams
            if 'wsd_'+DEM+'_b' not in GRASS_raster_files or force is True:
                gs.run_command('r.carve',raster=DEM,vector = v_stream, output= DEM+'_b', flags='n', width = 1*DEM_Res,depth=1),
                gs.run_command('g.remove', type ='raster', name=DEM, flags ='f')
            #rename burned DEM
            DEM = DEM+'_b'
                                   
        else:
            pass

        if 'wsd_'+DEM not in GRASS_raster_files or force is True:
            gs.run_command('g.copy',raster=DEM+',wsd_'+DEM)
            gs.run_command('g.remove', type ='raster', name=DEM, flags ='f')
        DEM='wsd_'+DEM
        
        print('Getting watershed information for '+huc12)
        #get accumulation and drainage direction for buffered area

        #flags for check
        area_check = 0
        attempt = -1
        adjustment_matrix = [0,10,100,-10,-100,-500]
        
        while area_check != 1:
            
            attempt+=1 #change buffer as needed to get out of dead areas
        
            #create a small buffer so that the outlet point is just outside of the HUC-12 boundary
            gs.run_command('v.buffer',input='tmp_huc',output='tmp_huc_sm_buff',distance = ((20+adjustment_matrix[attempt]) //DEM_Res))

            if draindir not in GRASS_raster_files or force is True:
                #get watershed statistics with buffer
                gs.run_command('r.watershed', elevation=DEM,drainage = draindir, accumulation = accum, flags= 'a',memory = max_mem)

            if draindir+'_tight' not in GRASS_raster_files or force is True:
                #mask to tighter area to get desired outlets
                gs.run_command('g.region',vector = 'tmp_huc_sm_buff',align=DEM)
                gs.run_command('r.mask',vector = 'tmp_huc_sm_buff')
                gs.run_command('r.watershed', elevation=DEM,drainage = draindir+'_tight', flags= 'a',memory = max_mem)


            #create outlets where flow directions are outside of the mask
            gs.run_command('r.mapcalc',overwrite=True,\
                      expression='outlets = if(%s >= 0,null(),1)' %(draindir+'_tight'))

            #get the accumulation at each potential outlet
            gs.run_command('r.mapcalc',overwrite=True,\
                      expression='outlets_accum = outlets * %s' %(accum))

            #get max accumulation at potential outfalls and print for testing
            max_accum = gs.parse_command('r.info',map = 'outlets_accum',flags='s')['max']
            #print(max_accum)

            #convert outlet with highest accumulation to raster 
            gs.run_command('r.mapcalc',overwrite=True,\
                      expression='outletmap = if(outlets_accum > %s,1,null())' %(float(max_accum)-1))
            #print(gs.parse_command('r.info',map = 'outletmap',flags='s'))
            gs.run_command('r.to.vect',overwrite=True,\
                          input = 'outletmap', output = 'outletsmapvec',\
                          type='point')

        
            #flags for check
            pulled_back_check = 0        
            print('Identifying outlet point for '+huc12)
            while pulled_back_check != 1:
                elevation_buffer = 10 #10 foot elevation offset as start. To be updated to be something more sophisticated with slope, etc.
            
                if huc12 != upstream_list[-1] and huc12 == outlet_keys[value]:
                    #mask out accomulation where elevations are within 10 meters of the outlet
                    #get elevation at outlet
                    gs.run_command('r.mapcalc',overwrite=True,\
                              expression='DEM_outlet = outletmap * %s' %(DEM))
                    outlet_elev = gs.parse_command('r.info',map = 'DEM_outlet',flags='s')['max']
                    #calculate 10m increase
                    mask_elev = float(outlet_elev) + elevation_buffer
                    gs.run_command('r.mapcalc',overwrite=True,\
                              expression='accum_pulled_back = if(%s >= %s,%s,null())' %(DEM,mask_elev,accum))

                    #find max accum 10m higher than outlet
                    max_accum_pulled_back = gs.parse_command('r.info',map = 'accum_pulled_back',flags='s')['max']
                    #print(max_accum)
                    
                    #convert outlet with highest accumulation to raster 
                    gs.run_command('r.mapcalc',overwrite=True,\
                              expression='outletmap = if(accum_pulled_back > %s,1,null())' %(float(max_accum_pulled_back)-1))
                    gs.run_command('r.to.vect',overwrite=True,\
                               input = 'outletmap', output = 'outletsmapvec_pulled_back',\
                               type='point')
                
                    #check that pulled back location is within 2500 ft of outlet
                    length_pb = gs.parse_command('v.distance', from_ ='outletsmapvec_pulled_back', to='outletsmapvec',upload='dist',flags='p')
                    pulled_back_dist = float(list(length_pb)[-1].split('|')[1])
                    if pulled_back_dist < 2500*0.3048:
                        gs.run_command('g.copy',vector='outletsmapvec_pulled_back,outletsmapvec')
                        pulled_back_check = 1
                    else:
                        elevation_buffer -= 2
                else:
                    pulled_back_check = 1
                    pass
        
                

            #export for testing
            gs.run_command('v.out.ogr', input=  'outletsmapvec' ,output = vector_dir/outlet_name,format = 'ESRI_Shapefile')

            print('Delineating basin according to outlet for '+huc12)
            #reset mask to larger buffer
            gs.run_command('g.region',vector = 'tmp_huc_buffer',align=DEM)
            gs.run_command('r.mask',vector = 'tmp_huc_buffer')
            gs.run_command('r.stream.basins',overwrite=True,\
                      direction=draindir,points='outletsmapvec',\
                      basins = basins)

            basin_size = float(gs.parse_command('r.info',map = basins,flags='s')['n'])*DEM_Res**2
            if basin_size > 0.5*huc_ws_area:
                area_check = 1
            else:
                pass

    
        gs.run_command('g.copy',raster=basins+',wsd_'+basins)
        
        print('Exporting basin as tiff') 
        #reset basin values to HUC12 value # and create dictionary key
        gs.run_command('r.mapcalc',overwrite=True,\
                  expression='%s = if(%s >= 0,%s,null())' %('out_'+basins,'wsd_'+basins,short_value))
        huc_key[short_value] = value
        
        #save and append raster name
        basin_list.append('out_'+basins)
        gs.run_command('g.remove', type ='raster', name=DEM, flags ='f')
        gs.run_command('g.remove', type ='raster', name=accum, flags ='f')
        
        #temp for testing
        gs.run_command('r.out.gdal', input= 'out_'+basins, output= raster_dir/gtiff_temp,createopt="BIGTIFF=YES,PROFILE=GeoTIFF,TFW=YES,COMPRESS=LZW")
        
    #reset region and patch basins from upstream to downstream
    print('Patching huc12s for export')
        #set the region based on the buffer but align with DEM raster
    DEM = basin_dems[huc12] #reset grid to the last used DEM grid
    gs.run_command('g.region',vector = upstream_huc12s+'_buffer',align=DEM)
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    if 'MASK' in GRASS_raster_files:
        gs.run_command('r.mask',flags = 'r')
    if len(basin_list) > 1:
        gs.run_command('r.patch',input = basin_list,output = 'compiled_basins')
        gs.run_command('r.to.vect',input = 'compiled_basins',output = 'v_basins', type = 'area')
    else:
        gs.run_command('r.to.vect',input = basin_list[0],output = 'v_basins', type = 'area')
    gs.run_command('v.clean',input = 'v_basins',threshold = 5000*DEM_Res,tool='rmarea',output = 'v_basins_cl')
    gs.run_command('v.out.ogr', input=  'v_basins_cl' ,output = vector_dir/basin_name, format = 'ESRI_Shapefile')
    
    return huc_key
    
def get_RAM():
    return psutil.virtual_memory().available / 1000000

def get_upstream_basin_areas(selected_huc,huc_type) -> dict:
    '''
    takes a HUC12 and then queries USGS HUC-12 services
    to identify all upstream HUC12s then gets basin areas and reformats based on huc_size.
    Use WWF function if not in the US
    '''
    import json
    #intialize lists
    huc12s = []
    #current USGS wbd service
    nhd_service = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/6/query?' #base url
    #set parameters for query
    where_c_init = "huc12 = '"+ selected_huc+"'"
    nhd_param = {'outFields':'*','where':where_c_init,
                 'f':'json','returnGeometry':'false'}
    #get query results
    print('searching for upstream huc12 basins')
    local_huc12 = esri_rest_query(nhd_service,nhd_param)
    #start crawling USGS for upstream HUCs
    if not local_huc12:
        return ['service error']
    if local_huc12['features']:
        huc12_upstream_target = local_huc12['features']
        while huc12_upstream_target:
            for huc12_upstream in huc12_upstream_target:
                #add huc12 to ruinning list, search for more upstream basins, remove from target list
                huc12s.append(huc12_upstream['attributes']['huc12'])
                huc12_upstream_target.remove(huc12_upstream)
                where_c = "tohuc = '"+ huc12_upstream['attributes']['huc12']+"'"
                nhd_param = {'outFields':'huc12','where':where_c,'f':'json','returnGeometry':'false'}
                huc12_upstream = esri_rest_query(nhd_service,nhd_param)['features']
                for target in huc12_upstream:
                    if target not in huc12_upstream_target:
                        huc12_upstream_target.append(target)
                
        print('found '+str(len(huc12s))+' upstream huc12s')
        return huc12s
        print('Not in USA')
        return ['not in USA']

def aoi_to_basin_shp(upstream_list,wbd_dataset,basin_scale,force=True):
    huc_key = {'HUC12':['WBDHU12',12], 'HUC10':['WBDHU10',10], 'HUC8':['WBDHU8',8]}
    GRASS_vector_files= [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
    
    selector = []
    for value in upstream_list:
        if value[:huc_key[basin_scale][1]] not in selector:
            selector.append(value[:huc_key[basin_scale][1]])
    if 'basins' not in GRASS_vector_files or force:
        if basin_scale+selector[0][:4] not in GRASS_vector_files:
            gs.run_command('v.import', input= wbd_dataset, layer = 'WBDHU'+str(huc_key[basin_scale][1]),  output= basin_scale+selector[0][:4])
        if len(selector) == 1:
            gs.run_command('v.extract', input = basin_scale+selector[0][:4],output = 'basins', where= "%s = '%s'" % (basin_scale,selector[0]))
            print("%s = %s" % (basin_scale,selector[0]))
        else:
            gs.run_command('v.extract', input = basin_scale+selector[0][:4],output = 'basins', where= "%s IN %s" % (basin_scale,str(tuple(selector))))
        print('created basins vector')
    else:
        print('basins already exist')
    return 'basins', selector

def aoi_to_basin_grass(aoi_shp,force=True):
    GRASS_vector_files= [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]

    if 'basins' not in GRASS_vector_files or force:
        gs.run_command('v.import', input= aoi_shp, output= 'basins')
        print('created basins vector')
    else:
        print('basins already exist')
    return 'basins'



def basin_dems(DEM_tiles,basins,basin_list,basin_scale,raster_dir,buffer = 0,fill_nulls = True, remove_tiles = False,export_tiff = True,force=True):
    basin_DEMs = []
    for basin in basin_list:
        dem_name = 'dem_'+str(basin)
        gtiff = dem_name+'.tif'
        
        if gtiff not in os.listdir(raster_dir) or force:
            if type(basin) == str:
                gs.run_command('v.extract', input =basins, where= "%s = '%s'" % (basin_scale,basin), output = 'tmp_bsn')
            else:
                gs.run_command('v.extract', input =basins, where= "%s = %s" % (basin_scale,basin), output = 'tmp_bsn')
            
            gs.run_command('v.buffer',input='tmp_bsn',output='tmp_bsn_buffer',distance = buffer)
               
            print("Patching together rasters for", basin)
            #set the region based on the buffer but align with DEM raster
            gs.run_command('g.region',vector = 'tmp_bsn_buffer',align=DEM_tiles[0])
            #mask outside cells
            gs.run_command('r.mask',vector = 'tmp_bsn_buffer')
            
            if len(DEM_tiles) > 1:
                gs.run_command('r.patch', input=DEM_tiles, output=dem_name)

            else:
                gs.run_command('g.rename', raster=(DEM_tiles[0],dem_name))
            
            if fill_nulls == True:
                gs.run_command('r.fillnulls', input = dem_name,output = 'dem_tmp', method = 'bilinear')
                gs.run_command('g.rename', raster=('dem_tmp',dem_name))

            #export to gdal
            if export_tiff == True:
                gs.run_command('r.out.gdal', input= dem_name, output= raster_dir/gtiff,createopt="BIGTIFF=YES,PROFILE=GeoTIFF,TFW=YES,COMPRESS=LZW",flags='f')
        
        print('created '+gtiff)
        basin_DEMs.append(dem_name)
    
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    
    #remove intermediate tiles option
    if remove_tiles:
        for tile in DEM_tiles:
            gs.run_command('g.remove', type ='raster', name=tile, flags ='f')
    if 'MASK' in GRASS_raster_files:
        gs.run_command('r.mask',flags='r')
    return basin_DEMs
    
# def create_smart_breaklines(DEM,DEM_Res,huc12)
#     #script TBD
#     print('Getting areas of curvature for '+huc12)
#     #narrow valleys
#     gs.run_command('r.param.scale',input=DEM,output='curve_DEM_s',size='5',method='crosc')
#     #broader valleys
#     gs.run_command('r.param.scale',input=DEM,output='curve_DEM_m',size='11',method='crosc')
#     #broadest valleys
#     gs.run_command('r.param.scale',input=DEM,output='curve_DEM_lv',size='15',method='crosc')
#     #merge
#     gs.run_command('r.mapcalc',overwrite=True,\
#       expression='weight_%s = if(curve_DEM_s < 0, -100 * curve_DEM_s, if(curve_DEM_m < 0, -100 * curve_DEM_m, if(curve_DEM_lv < 0, -100 * curve_DEM_lv, 0.01)))' %(DEM))
#     gs.run_command('r.thin',input = 'weight_'+huc12+'_int', output = r_stream)
#     gs.run_command('r.to.vect',input=r_stream,output = v_stream,type = 'line')

def outlet_key(basin_scale,upstream_list):
    dem_dict = {'HUC8':8,'HUC10':10,'HUC12':12}
#     upstream_list.reverse() include if not already part of larger function
    ds_basins = {}
    for basin in upstream_list:
        at_scale = basin[:dem_dict[basin_scale]]
        ds_basins[at_scale] = basin
    return ds_basins

####Joint probability code developed by Nada Elgamal


def prep_huc12_gpd(vector_dir, trg_huc, crs):
    hdf_nm = 'NHDPLUS_H_'+str(trg_huc[:4])+'_HU4_GDB.gdb'
    huc_gdb = gpd.read_file(pl.Path(vector_dir)/hdf_nm, driver='FileGDB', layer='WBDHU12')
    huc_gdf = huc_gdb.to_crs(epsg=crs)
    huc_gdf['AreaSqMi_UTM'] = huc_gdf.geometry.area / 2589990 #conversion from meters squared to sq miles
    huc_gdf_cols = [col.lower() for col in huc_gdf.columns.tolist()]
    huc_gdf.columns = huc_gdf_cols    
    out_df_cols = ['TrgHUC12','TrgHUC12_SqMi','UpHUC12','UpHUC12_SqMi','UpTrgAr_SqMi','TotUpAr_SqMi','Drain_RA']
    out_df = pd.DataFrame(columns=out_df_cols)
    return huc_gdf, out_df

def getUpstreamHUC12(trg_huc,huc_gdf,out_df):
    if huc_gdf[(huc_gdf['huc12']==trg_huc) | (huc_gdf['tohuc']==trg_huc)].empty==False:
        if huc_gdf[huc_gdf['tohuc']==trg_huc].empty==False:
            uphuc_list = huc_gdf[huc_gdf['tohuc']==trg_huc].huc12.tolist()
            trg_huc_ar = huc_gdf[huc_gdf['huc12']==trg_huc].areasqmi_utm.values[0] 
#             print ("Target HUC: ", trg_huc, "\nTarget HUC area: ", round(trg_huc_ar,2), "\nUpstream HUCS: ", uphuc_list)
            
            for uphuc in uphuc_list:
#                 print ( "Immediate Upstream Basin:  ", uphuc)
                upstrm_hucs_ar, uphuc_out_df = getUpstreamHUC12(trg_huc,huc_gdf,out_df)

                uphuc_ar = huc_gdf[huc_gdf['huc12']==uphuc].areasqmi_utm.values[0] 
                uphuc_trg_ar = uphuc_ar + trg_huc_ar  # remove column later
                tot_uphuc_ar = trg_huc_ar + upstrm_hucs_ar
                drain_ratio = tot_uphuc_ar / trg_huc_ar

#                 print("Upstream HUC area: ", uphuc_ar)        
                out_df = uphuc_out_df.append({'TrgHUC12':trg_huc, 'TrgHUC12_SqMi':trg_huc_ar,
                                              'UpHUC12':uphuc, 'UpHUC12_SqMi':uphuc_ar, 
                                              'UpTrgAr_SqMi':uphuc_trg_ar, 'TotUpAr_SqMi':tot_uphuc_ar, 'Drain_RA': drain_ratio}, ignore_index=True)

        else:
            tot_uphuc_ar = huc_gdf[(huc_gdf['huc12']==trg_huc)].areasqmi_utm.values[0]
#             print("\tNo basins upstream of {}".format(trg_huc))
    else:
        tot_uphuc_ar = print("This HUC12 Watershed Has No Upstream Basins")
    
    out_df = out_df.sort_values('TrgHUC12', ascending=False).reset_index(drop=True)
    return tot_uphuc_ar, out_df

def Scale_HUC (out_scale, huc_df):
    huc_scale = int(out_scale.split('HUC').pop())
    huc_df['Up_HUC'] = huc_df['UpHUC12'].str[:huc_scale]
    uphuc_list = huc_df['Up_HUC'].unique().tolist()

    huc_dict={}

    for huc in uphuc_list:
        uphuc_df = huc_df[huc_df['UpHUC12'].str[:huc_scale].isin([huc])==True]
        uphuc_sr = uphuc_df.groupby(['Up_HUC']).UpHUC12_SqMi.sum()
        huc_ar = uphuc_sr.values

        for ix in uphuc_sr.index:
            TrgHUC12_list = uphuc_df.TrgHUC12.tolist()
            UpHUC12_list = uphuc_df.UpHUC12.tolist()
            trg_basin_ls = set([basin for basin in TrgHUC12_list if basin not in UpHUC12_list and huc in basin])

            for trg_basin in trg_basin_ls:
                if ix in trg_basin[:huc_scale]:
                    trg_ar = uphuc_df[uphuc_df['TrgHUC12']==trg_basin].TrgHUC12_SqMi.drop_duplicates()
                    huc_ar = trg_ar.values + uphuc_sr.values

            huc_dict[huc] = huc_ar

    drain_ar_df = pd.DataFrame.from_dict(huc_dict, orient='index', columns=['Drainage_Ar_SqMi'])
    
    return drain_ar_df


def grass_watershed_processing_dev(basin_scale,upstream_list,upstream_huc12s,upstream_nhd,vector_dir,raster_dir,filter_size = '23', carve = False, force=False):
    '''get accumulation and drain direction, create streams, snap outlets delineate
    post processing of watersheds done separately'''
    
    #assign list of dems:
    basin_dems = dem_assign(basin_scale,upstream_list)

    #get full region with buffer to set initial project bounds
    #create a 2000 meter buffer around the HUC-12 to account for differences between the high resolution DEM and NED
    set_full_bounds(upstream_huc12s,2000,basin_dems[upstream_list[0]])
    
    #Get DEM resolution
    DEM_Res = get_dem_res(dem_name)
    
    #set variables
    Stream_Threshold_Local= 2590000/DEM_Res #1 sq miles   
    basin_list = []
    huc_key = {}
    if float(upstream_list[0]) > float(upstream_list[-1]):
        upstream_list.reverse()
    outlet_keys = outlet_key(basin_scale,upstream_list)
    basin_name = 'v_domains.shp'
    
    #perform grass processing
    for huc12 in upstream_list:  
        #initiate variables
        value = huc12[:dem_dict[basin_scale]] #trim value to desired huc level
        short_value = huc12[-7:dem_dict[basin_scale]] #small enough to retain int raster status
        GRASS_vector_files= [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
        GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
        tmp_huc = 'tmp_huc'
        draindir = 'r_drain_dir_'+huc12
        outlet_r = 'r_outlet_'+huc12
        accum = 'r_accum_'+huc12
        out_basins = 'wsd_basin_'+huc12
        r_stream = 'stream_'+huc12
        v_stream = 'v_stream_'+huc12
        v_outlet = 'v_outlet_'+huc12
        v_basins = 'v_'+basins  
        outlet_name ='v_'+huc12+'_outlet.shp'
        DEM = basin_dems[huc12]
        gtiff_temp = basins+'.tif'
        
        #skip functions if already delineated
        if os.path.exists(raster_dir/gtiff_temp): 
            print('data already created for',huc12)
            basin_list.append('out_'+basins)
            huc_key[short_value] = value
            continue
        
        #extract huc12 area of interest
        gs.run_command('v.extract', input =upstream_huc12s, where= "HUC12='%s'" % huc12, output = tmp_huc)
          
        #calculate the approximate watershed area 
        ws_calc = gs.parse_command('v.to.db', map=tmp_huc, option='area',flags='pc',columns ='area')
        huc_ws_area = float(list(ws_calc)[-1].split('|')[1])
               
        #get memory for watershed scripts
        max_mem = get_RAM()
        
        #create buffers for area
        buffer_full, buffer_95_pct = create_buffers_and_mask(tmp_huc,DEM)
        
        #apply median filter
        DEM = median_filter(huc12,DEM,filter_size,force)
        
        #extract area specific nhd lines
        if 'nhd_'+'r_accum'+huc12 not in GRASS_raster_files or force is True:
            #extract nhd for area of interest
            gs.run_command('v.overlay', ainput = upstream_nhd,binput = tmp_huc+'_buffer_nhd', output = v_stream, operator = 'and',flags='t')

        if v_stream not in GRASS_vector_files or force is True:
            gs.run_command('v.out.ogr',input=v_stream,output=vector_dir,type= 'line',format='ESRI_Shapefile')
        
        #apply carving for NHD lines
        if 'wsd_'+DEM+'_b' not in GRASS_raster_files or force:
            DEM = carve_dem(vector_area,v_stream,DEM,force)
            
        #get grass streams for secondary carving
        print('Getting watershed information for '+huc12)
        gs.run_command('r.watershed', elevation=DEM,accumulation = accum, flags= 'a',memory = max_mem)
        gs.run_command('r.stream.extract', elevation=DEM, accumulation = accum, threshold =Stream_Threshold_Local, stream_raster = r_stream+'_grass',stream_vector = v_stream+'_grass',memory = max_mem)
        DEM = carve_dem(vector_area,v_stream+'_grass',DEM,force)
        
        print('Performing watershed delineation for '+huc12)
        #get accumulation and drainage direction for buffered area

        #flags for check
        area_check = 0
        attempt = -1
        adjustment_matrix = [2,10,100,-10,-100,-500]
        
        while area_check != 1:
            attempt+=1 #change buffer as needed to get out of dead outlet zones
            get_outlets(tmp_huc,attempt,adjustment_matrix,DEM_Res,draindir,accum,max_mem)
            
            #flags for check
            pulled_back_check = 0        
            print('Identifying outlet point for '+huc12)
            elevation_buffer = 10 #10 foot elevation offset as start. To be updated to be something more sophisticated with slope, etc.
            while pulled_back_check != 1:
                if huc12 != upstream_list[-1] and huc12 == outlet_keys[value]:
                    pulled_back_check, elevation_buffer = pulled_back_delineation(elevation_buffer,DEM,accum)
                else:
                    pulled_back_check = 1
                    pass

            #export for testing
            gs.run_command('v.out.ogr', input=  'outletsmapvec' ,output = vector_dir/outlet_name,format = 'ESRI_Shapefile')
            print('Delineating basin according to outlet for '+huc12)
            basin_size = delineate_basins(tmp_huc,draindir,DEM)

            if basin_size > 0.5*huc_ws_area:
                area_check = 1
            else:
                pass
        
        print('Exporting basin as tiff') 
        #reset basin values to HUC12 value # and create dictionary key
        gs.run_command('r.mapcalc',overwrite=True,\
                  expression='%s = if(%s >= 0,%s,null())' %('out_'+basins,out_basins,short_value))
        huc_key[short_value] = value
        
        #save and append raster name
        basin_list.append('out_'+basins)
        gs.run_command('g.remove', type ='raster', name=DEM, flags ='f')
        gs.run_command('g.remove', type ='raster', name=accum, flags ='f')
        
        #temp for testing
        gs.run_command('r.out.gdal', input= 'out_'+basins, output= raster_dir/gtiff_temp,createopt="BIGTIFF=YES,PROFILE=GeoTIFF,TFW=YES,COMPRESS=LZW")
        
    #reset region and patch basins from upstream to downstream
    print('Patching huc12s for export')
        #set the region based on the buffer but align with DEM raster
    DEM = basin_dems[huc12] #reset grid to the last used DEM grid
    gs.run_command('g.region',vector = upstream_huc12s+'_buffer',align=DEM)
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    if 'MASK' in GRASS_raster_files:
        gs.run_command('r.mask',flags = 'r')
    gs.run_command('r.patch',input = basin_list,output = 'compiled_basins')
    gs.run_command('r.to.vect',input = 'compiled_basins',output = 'v_basins', type = 'area')
    gs.run_command('v.clean',input = 'v_basins',threshold = 5000*DEM_Res,tool='rmarea',output = 'v_basins_cl')
    gs.run_command('v.out.ogr', input=  'v_basins_cl' ,output = vector_dir/basin_name, format = 'ESRI_Shapefile')
    
    return huc_key

def median_filter(dem,filter_size,force):
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    DEM_Res = get_dem_res(DEM)
    if 'filt_'+dem not in GRASS_raster_files or force:
        gs.run_command('r.neighbors', input = dem,output='filt_'+dem,method='median',size=filter_size//DEM_Res)
    return 'filt_'+dem


def carve_dem(v_stream,DEM,force):
    '''applys carve function based on vector area, vector lines and dem
    '''
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    GRASS_vector_files= [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
    DEM_Res = get_dem_res(DEM)
    #carving with streams
    gs.run_command('r.carve',raster=DEM,vector = v_stream, output= DEM+'_b', flags='n', width = 1*DEM_Res,depth=1),
#     gs.run_command('g.remove', type ='raster', name=DEM, flags ='f')
    #rename burned DEM
    return DEM+'_b'



def create_buffers_and_mask(area_vector,DEM):
    #create a 2000 meter buffer around the HUC-12 to account for differences between the high resolution DEM and NED
    gs.run_command('v.buffer',input= area_vector,output=area_vector+'_buffer',distance = 2000)

    #create a slightly smaller buffer around the HUC-12 to avoid carving null cells
    gs.run_command('v.buffer',input=area_vector,output=area_vector+'_buffer_nhd',distance = 2000*0.95)

    #set the region based on the buffer but align with DEM raster
    gs.run_command('g.region',vector = area_vector+'_buffer',align=DEM)

    #mask outside cells
    gs.run_command('r.mask',vector = area_vector+'_buffer')
    
    return area_vector+'_buffer', area_vector+'_buffer_nhd'

def dem_assign(basin_scale,upstream_list):
    basin_dems = {}
    dem_dict = {'HUC8':8,'HUC10':10,'HUC12':12,'USER DEFINED':None}
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    for basin in upstream_list:
        dem_name = 'dem_'+basin[:dem_dict[basin_scale]]
        if dem_name in GRASS_raster_files: 
            basin_dems[basin] = dem_name
        else:
            print(dem_name,'not in Grass raster files')
    return basin_dems

def set_full_bounds(vector,buffer_dist,dem_name):
    gs.run_command('v.buffer',input=vector,output=vector+'_buffer',distance = buffer_dist)
    
    #set the region based on the buffer but align with DEM raster
    gs.run_command('g.region',vector = vector+'_buffer',align=dem_name)
    
def get_dem_res(dem_name):
    meta_dem = gs.parse_command('r.info',map=dem_name,flags='g')
    return float(meta_dem['nsres'])

def get_outlets(tmp_huc,attempt,adjustment_matrix,DEM,DEM_Res,draindir,accum,max_mem,force=True):
    GRASS_vector_files= [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    #create a small buffer so that the outlet point is just outside of the HUC-12 boundary
    gs.run_command('v.buffer',input=tmp_huc,output=tmp_huc+'_sm_buff',distance = ((20+adjustment_matrix[attempt]) //DEM_Res))

    if draindir not in GRASS_raster_files or force is True:
        #get watershed statistics with buffer
        gs.run_command('g.region',vector = tmp_huc+'_buffer',align=DEM)
        gs.run_command('r.mask',vector = tmp_huc+'_buffer')
        gs.run_command('r.watershed', elevation=DEM,drainage = draindir, accumulation = accum, flags= 'a',memory = max_mem)

    if draindir+'_tight' not in GRASS_raster_files or force is True:
        #mask to tighter area to get desired outlets
        gs.run_command('g.region',vector = tmp_huc+'_sm_buff',align=DEM)
        gs.run_command('r.mask',vector = tmp_huc+'_sm_buff')
        gs.run_command('r.watershed', elevation=DEM,drainage = draindir+'_tight', flags= 'a',memory = max_mem)


    #create outlets where flow directions are outside of the mask
    gs.run_command('r.mapcalc',overwrite=True,\
              expression='outlets = if(%s >= 0,null(),1)' %(draindir+'_tight'))

    #get the accumulation at each potential outlet
    gs.run_command('r.mapcalc',overwrite=True,\
              expression='outlets_accum = outlets * %s' %(accum))

    #get max accumulation at potential outfalls and print for testing
    max_accum = gs.parse_command('r.info',map = 'outlets_accum',flags='s')['max']
    #print(max_accum)

    #convert outlet with highest accumulation to raster 
    gs.run_command('r.mapcalc',overwrite=True,\
              expression='outletmap = if(outlets_accum > %s,1,null())' %(float(max_accum)-1))
    #print(gs.parse_command('r.info',map = 'outletmap',flags='s'))
    gs.run_command('r.to.vect',overwrite=True,\
                  input = 'outletmap', output = 'outletsmapvec',\
                  type='point')

def pulled_back_delineation(elevation_buffer,DEM,accum):
    #mask out accomulation where elevations are within 10 meters of the outlet
    #get elevation at outlet
    gs.run_command('r.mapcalc',overwrite=True,\
              expression='DEM_outlet = outletmap * %s' %(DEM))
    outlet_elev = gs.parse_command('r.info',map = 'DEM_outlet',flags='s')['max']
    #calculate 10m increase
    mask_elev = float(outlet_elev) + elevation_buffer
    gs.run_command('r.mapcalc',overwrite=True,\
              expression='accum_pulled_back = if(%s >= %s,%s,null())' %(DEM,mask_elev,accum))

    #find max accum 10m higher than outlet
    max_accum_pulled_back = gs.parse_command('r.info',map = 'accum_pulled_back',flags='s')['max']
    #print(max_accum)

    #convert outlet with highest accumulation to raster 
    gs.run_command('r.mapcalc',overwrite=True,\
              expression='outletmap = if(accum_pulled_back > %s,1,null())' %(float(max_accum_pulled_back)-1))
    gs.run_command('r.to.vect',overwrite=True,\
               input = 'outletmap', output = 'outletsmapvec_pulled_back',\
               type='point')

    #check that pulled back location is within 2000 ft of outlet
    length_pb = gs.parse_command('v.distance', from_ ='outletsmapvec_pulled_back', to='outletsmapvec',upload='dist',flags='p')
    pulled_back_dist = float(list(length_pb)[-1].split('|')[1])
    if pulled_back_dist < 2000:
        gs.run_command('g.copy',vector='outletsmapvec_pulled_back,outletsmapvec')
        pulled_back_check = 1
        return pulled_back_check, elevation_buffer
    else:
        elevation_buffer -= 2
        return pulled_back_check, elevation_buffer

def delineate_basins(tmp_huc,draindir,DEM,out_basins):
    #reset mask to larger buffer
    gs.run_command('g.region',vector = tmp_huc+'_buffer',align=DEM)
    gs.run_command('r.mask',vector = tmp_huc+'_buffer')
    gs.run_command('r.stream.basins',overwrite=True,\
              direction=draindir,points='outletsmapvec',\
              basins = out_basins)

    basin_size = float(gs.parse_command('r.info',map = out_basins,flags='s')['n'])
    return basin_size

def plot_accumulation(array, 
                      out_path, 
                      legend_label="Number of upstream cells", 
                      plot_title="Flow accumulation", 
                      cmap="cubehelix"):
    fig, ax = plt.subplots(figsize=(15,15))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    
    im = ax.imshow(array, zorder=1, cmap=cmap, norm=colors.LogNorm(1,np.nanmax(array)))
    plt.colorbar(im, ax=ax, label=legend_label)
    plt.title(plot_title)
    plt.savefig(out_path)

def plot_grass_layer(layer, file_path, vector=False):
    """ 
    Plot a specific grass layer to a file. 
    Note the watershed boundary + buffered boundary should be defined as 
    hu and hucbuff respectively
    """
    
    os.environ["GRASS_RENDER_FILE"] = str(file_path)
    gs.run_command('d.erase')
    if vector:
        try:
            gs.run_command('r.to.vect', input=layer, output=layer+"_v", type="line")
            gs.run_command('d.vect', map=layer+"_v")
        except:
            gs.run_command('d.vect', map=layer)
    else:
        gs.run_command('d.rast', map=layer)
        gs.run_command('d.legend', raster=layer)
    try:
        gs.run_command('d.vect', map="huc", type="boundary", color="black", width="3")
        gs.run_command('d.vect', map="hucbuff", type="boundary", color="black", width="1")
    except:
        logger.warning(f"Couldn't find watershed boundary and/or buffered watershed boundary!")
        
def import_watershed_vectors(config,overwrite = False):
    GRASS_vector_files = [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
    gs.run_command('g.region',raster=config.dem.name,align=config.dem.name)
    if 'huc' in GRASS_vector_files and overwrite is False:
        logger.info("not reimporting")
    else:
        logger.info("importing huc boundary")
        gs.run_command(
            'v.import', 
            input=config.root/config.huc_bounds.out_file.geojson, 
            output="huc", 
            overwrite = True
        ) 

        logger.info("setting up buffer around boundary")

        gs.run_command(
            'v.buffer', 
            input="huc", 
            output="hucbuff",
            distance=config.huc_buff # distance in meters
        ) 
        #create buffer within full buffer
        gs.run_command(
        'v.buffer', 
        input="huc", 
        output="hucbuff_95pct",
        distance=config.huc_buff*0.95 # distance in meters
        ) 

        logger.info("importing NHD")
        gs.run_command(
            'v.import', 
            input=config.root/config.nhd.out_file.geojson, 
            output="tmp_flowline", 
            extent="region", overwrite = True
        ) 
        gs.run_command(
            'v.extract', 
            input='tmp_flowline', 
            output="tmp_flowline_query", 
            where="FCode <> {}".format(56600), overwrite = True
        ) 
        #clipping at 95% to avoid pulling in nulls
        gs.run_command(
        'v.clip', 
        input="tmp_flowline_query",
        clip='hucbuff_95pct',
        output="nhd_flowline", 
        overwrite = True
        )
        gs.run_command(
        'g.remove', 
        type='vector',
        name = 'tmp_flowline',
        flags='f'
        )
    config.huc_bounds.name = 'huc'
    config.huc_bounds.buffer = 'hucbuff'
    config.nhd.name = 'nhd_flowline'
    return config


                              
def dem_to_grass(config,overwrite = False):
    dem_out_vrt = config.root/config.dem.out_file.vrt
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    """imports rasters to given grass session in gs"""
    raster_name = ('%s' % pl.Path(config.dem.out_file.vrt).name)[:-4]
    config.dem.name = raster_name
    if raster_name in GRASS_raster_files and overwrite is False:
        logger.info('DEM already in GRASS. Only re-exporting the tif')
        pass
    else:
        gs.run_command(
            'r.import', 
            input=config.root/config.dem.out_file.vrt, 
            output=raster_name, 
            overwrite=True, 
            memory = config.mem
        )

        gs.run_command('g.region', raster=raster_name,align=raster_name)
        logger.info('Checking vertical units and converting to :' + config.vert_unit)
        dem_vert_unit_check(dem_out_vrt,raster_name,config.vert_unit)
        #fill nulls
        gs.run_command('r.fill.stats', input = raster_name,output = 'dem_tmp', distance=3,mode='wmean',power=5,cells=8,flags='k')
        gs.run_command('g.rename', raster= "dem_tmp,{}".format(raster_name))

    #Get DEM resolution 
    meta_dem = gs.parse_command('r.info',map=raster_name,flags='g')
    DEM_Res = float(meta_dem['nsres'])
    import_watershed_vectors(config,overwrite)
    mask_dem(config)
    plot_grass_layer(raster_name, str(config.log.out/"bare_earth_dem.png"), vector=False)
    if config.dem.out_file.tiff:
        gs.run_command('r.out.gdal', input= raster_name, output= config.root/config.dem.out_file.tiff,createopt="BIGTIFF=YES,PROFILE=GeoTIFF,TFW=YES,COMPRESS=LZW")
        logger.info('Exported bare earth tiff: {}'.format(config.dem.out_file.tiff))
    return config



def roads_detector(df_pts,ratio):
    #roads and their average width in USA is 9 - 15 m (Hughes et al.,2004). Reccomends median filter of 21m
    #assume 2 foot max delta between min and max per of rise per 10 meters of lenth -
    #is road if len / (maxZ - minZ) < 5, else not 
    distance = df_pts['distance'].sum()
    if distance < 500:
        height = df_pts['z'].max() - df_pts['z'].min()
        if distance / height <= ratio:
            return df_pts, distance / height
        else:
            return pd.DataFrame(columns = df_pts.columns.to_list()), 0
    else:
        return pd.DataFrame(columns = df_pts.columns.to_list()), 0

def create_carve_lines(config,overwrite = False):
    config = create_draft_stream_accum(config,overwrite = True)
    create_3d_nhd_lines(config,overwrite = False)
    reset_nhd_elevations_with_dem(config,overwrite = False)
    create_hydro_connectors(config,overwrite = False)
    config = topo_carve(config,overwrite = False)
    return config


def develop_watershed_data(config, overwrite = False):
     #calculate watershed properties (accumulation, drainage direction, basins)
    # for the burned DEM 
    
    #set region and get cell size
    reg = gs.parse_command('g.region', 
                           raster=config.dem.enforced, 
                           flags='pgm', 
                           zoom=config.dem.enforced,
                           align=config.dem.enforced
                          )
    cell_size = float(reg.nsres)
    
    #get dem name
    dem = config.dem.name
    #create new variables
    #create new variables
    accum = dem[:dem.find('dem')]+'accumulation'
    drain_dir = dem[:dem.find('dem')]+'drain_dir'
    v_streams = dem[:dem.find('dem')]+'stream_v'
    r_streams = dem[:dem.find('dem')]+'stream_r'
    v_basins = dem[:dem.find('dem')]+'basin_v'
    r_basins = dem[:dem.find('dem')]+'basin_r'
    
    gs.run_command(
        'r.watershed', 
        elevation= config.dem.enforced, threshold=config.thresholds['stream'] // cell_size**2, 
        accumulation=accum, drainage = drain_dir,  
        flags='sab', overwrite=True, memory = config.mem
    )
    logger.info("plotting flow accumulation")
    accum_array = mask_array(garray.array('MASK'), garray.array(accum))
    plot_accumulation(accum_array, config.log.out/"accumulation.png")
    gs.run_command("r.stream.extract", 
                   elevation= config.dem.enforced,   
                   threshold=config.thresholds['stream'] // cell_size**2,
                   accumulation= accum, 
                   stream_raster = r_streams,
                   stream_vector = v_streams,
                   memory = config.mem
                  )
    plot_grass_layer(v_streams, str(config.log.out/"streams.png"), vector=True)
    #adding variables to config file
    config.accum = accum
    config.drain = drain_dir
    #export for GIS
    out_dir = config.root/'outputs'
    # if os.path.exists(out_dir) and os.path.isdir(out_dir):
    #     shutil.rmtree(out_dir)
    # os.makedirs(out_dir)
    config.out_dir = out_dir
    logger.info('Exporting streams lines to : {}'.format(out_dir))
    gs.run_command('v.out.ogr',input=v_streams,output=out_dir/'{}.gpkg'.format(v_streams),type= 'line',format='GPKG')
    logger.info('Exporting accumulation raster to {}'.format(out_dir))
    gs.run_command('r.out.gdal', input= accum, 
                   output= config.out_dir/'{}.tif'.format(accum),
                   createopt="BIGTIFF=YES,PROFILE=GeoTIFF,TFW=YES,COMPRESS=LZW"
                  )
    logger.info('Exporting drainage direction raster to : {}'.format(out_dir))
    gs.run_command('r.out.gdal', input= drain_dir, 
                   output= config.out_dir/'{}.tif'.format(drain_dir),
                   createopt="BIGTIFF=YES,PROFILE=GeoTIFF,TFW=YES,COMPRESS=LZW"
                  )
    #delineate watersheds
    pour_points = True if config.pour_points.in_file != 'None' else False
    if pour_points:
        GRASS_vector_files = [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
        if 'pour_points' in GRASS_vector_files and overwrite is False:
            logger.info("not reimporting")
        else:
            logger.info("importing basin pour points")
            gs.run_command(
                'v.import', 
                input=config.root/config.pour_points.in_file,
                extent='region',
                output="pour_points", 
                overwrite = True
            ) 
        gs.run_command("r.stream.snap", 
                       input= 'pour_points',stream_rast=r_streams,
                   output ='{}_snapped'.format('pour_points'),
                  radius=(50), accumulation= accum, threshold = (config.thresholds['stream'] // cell_size**2)*10,
                       memory = config.mem)
        gs.run_command("r.stream.basins", 
                       direction = drain_dir,
                       basins =r_basins,
                       points ='{}_snapped'.format('pour_points'),
                       memory = config.mem)

        gs.run_command('r.to.vect',input = r_basins,output = v_basins, type = 'area')
        gs.run_command('v.clean',input = v_basins,threshold = 5000*cell_size,tool='rmarea',output = '{}_clean'.format(v_basins))
        logger.info('Exporting accumulation raster to : {}'.format(out_dir))
        plot_grass_layer('{}_clean'.format(v_basins), str(config.log.out/"basins.png"), vector=True)
        
        gs.run_command('v.out.ogr', input=  '{}_clean'.format(v_basins) ,type = 'area',output = out_dir/'{}.gpkg'.format(v_basins), format = 'GPKG')
    else:
        #delineate based on DEM outlet
        # basin_delineation_from_dem
        pass
    return config

def layer_indexer_all(url,params):
    layers = []
    r = requests.get(url,params,verify=False)
    rest_layers = json.loads(r.content)['layers']
    for l in rest_layers:
        layers.append(l['id'])
    return layers

def get_dam_names(esri_geo,layers,out_field):
    api_base = 'https://ags03.sec.usace.army.mil/server/rest/services/Dams_Public/FeatureServer/0'
    query_params = {'geometry': esri_geo, 'geometryType': 'esriGeometryPolygon','f':'json','outFields':out_field,'returnGeometry':'false'}
    response = esri_rest_query(query_url,param)
    if 'features' in response.keys():
        if response['features']:
            return response['features'][0]['NAME']
        else:
            return 'None'
            
def basin_delineation_from_dem():
    return None

   

def create_draft_stream_accum(config,overwrite = False):
    #get dem name
    dem = config.dem.name
    out_dir = config.root/'outputs'
    #create new variables
    accum = dem[:dem.find('dem')]+'accumulation'
    stream_rast = dem[:dem.find('dem')]+'stream_r'
    v_basins = dem[:dem.find('dem')]+'basin_v_draft'
    r_basins = dem[:dem.find('dem')]+'basin_r_draft'
    logger.info("performing initial pass of watershed analysis")
    reg = gs.parse_command('g.region', raster=dem, flags='pgm', zoom=dem,align=dem)
    cell_size = float(reg.nsres)
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    if stream_rast in GRASS_raster_files and overwrite is False:
        logger.info('watershed data already in GRASS. Skipping')
    else:
        gs.run_command(
            'r.watershed', 
            elevation=dem, threshold=config.thresholds['stream'] // cell_size**2, 
            accumulation=accum,
            stream = stream_rast,
            flags='sab', overwrite=True, memory = config.mem
        )
    # gs.run_command('r.to.vect',input = r_basins,output = v_basins, type = 'area')
    # gs.run_command('v.clean',input = v_basins,threshold = 5000*cell_size,tool='rmarea',output = '{}_clean'.format(v_basins))
    # gs.run_command('v.out.ogr', input=  '{}'.format(v_basins) ,type = 'area',output = out_dir/'{}.gpkg'.format(v_basins), format = 'GPKG')
    config.accum = accum
    config.rstream = stream_rast
    return config

def adjust_nhd_lines(config,overwrite = False):
    logger.info("combining watershed lines and nhd lines")
    #snap points to accum lines for more accurate streams
    dem = config.dem.name
    nhd = config.nhd.name
    stream_rast = config.rstream
    gs.run_command("v.build.polylines",input=nhd,output= '{}_poly'.format(nhd),cats='first')
    radius = 4 #meters
    gs.run_command("v.to.points", input= '{}_poly'.format(nhd), type='line', output='{}_pts'.format(nhd),dmax=5)
    # gs.run_command("r.stream.snap", input= '{}_pts'.format(nhd),stream_rast=stream_rast,
    #                    output ='{0}_pts_snapped'.format(nhd),
    #                   radius=radius, memory = config.mem)
    # gs.run_command('v.db.connect',map='{}_pts_snapped'.format(nhd),table='{}_pts'.format(nhd))
    logger.info("capturing elevations along nhd lines")
    # gs.run_command('v.drape',input='{}_pts_snapped'.format(nhd),elevation=config.dem.name,type='point',output='{}_pts_3d'.format(nhd))
    gs.run_command('v.drape',input='{}_pts'.format(config.nhd.name),elevation=config.dem.name,type='point',output='{}_pts_3d'.format(config.nhd.name))


def get_road_names(esri_geo,sr,out_field):
    #set base URL information for road name check
    api_base = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Transportation/MapServer'
    params = {'f':'json'}
    # layer_index = layer_indexer_all(api_base,params)
    layer_index = [8]
    query_params = {'geometry': esri_geo, 'geometryType': 'esriGeometryPolyline','f':'json','outFields':out_field,'inSR':sr,'returnGeometry':'false','distance':'25'}
    for layer in layer_index:
        query_url = '{0}/{1}/query'.format(api_base,layer)
        response = esri_rest_query(query_url,query_params)
        if 'features' in response.keys():
            if response['features']:
                print(response['features'][0]['attributes']['NAME'])
                return response['features'][0]['attributes']['NAME']
            else:
                pass
        else:
            pass
    return 'None'

def esri_rest_query(service:str,parameters:dict) -> list:
    '''
    takes a url and parameters to 
    query arcGIS REST service for information
    SSL bypassed because of 1 case of verification
    error. This will be updated if an alternative source can be located
    '''
    gcontext = ssl.SSLContext()  # Needed to bypass SSL verification issue
    data = urllib.parse.urlencode(parameters)
    data = data.encode('ascii') # data should be bytes
    req = urllib.request.Request(service, data)
    with urllib.request.urlopen(req,context=gcontext) as response:
        projects = json.loads(response.read())
    return projects

def create_hydro_connectors(config,overwrite = False):
    dem = config.dem.name
    hydro_connectors = dem[:dem.find('dem')]+'hydro_connectors'
    out_dir = config.root/'working_points'
    GRASS_vector_files= [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
    if  hydro_connectors in GRASS_vector_files and overwrite is False:
        logger.info("hydroconnectors exist. Not recreating.")
    else:
        #clean up existing files
        if os.path.exists(out_dir) and os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir)
        gs.run_command('g.remove', type ='vector', name=hydro_connectors, flags ='f')

        logger.info("exporting nhd lines for road and dam analysis")

        cats = list(gs.parse_command('v.db.select',map='{}_pts_3d'.format(config.nhd.name),columns='cat',flags='c',separator=',').keys())
        for cat in cats:
            gs.run_command('v.out.ascii',flags='c',type='point',input ='{}_pts_3d'.format(config.nhd.name), cats= cat, output=out_dir/'nhd_out_{}.txt'.format(cat),format='wkt')
            
            points = []
            df_points = pd.DataFrame()
            with open(out_dir/"nhd_out_{}.txt".format(cat), 'r') as fd:
                txt = fd.read()
            points = txt.splitlines()
            df_points = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(points, index=None, crs=int(config.sr)))    
            temp = get_increases_along_streams(df_points.geometry.to_list(),cat,out_dir,int(config.sr),ratio=15)
        geojsons = []
        logger.info("importing carve lines for roads and dams")
        for file in os.listdir(out_dir):
                GRASS_vector_files= [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
                if file.endswith(".geojson"):
                    if hydro_connectors not in GRASS_vector_files:
                        gs.run_command('v.in.ogr',input=out_dir/file, output=hydro_connectors)
                    else:
                        gs.run_command('v.in.ogr',input=out_dir/file, output=file.split('.')[0])
                        gs.run_command('v.patch',flags = 'ae',input=file.split('.')[0],output=hydro_connectors)

                    # gs.run_command('g.remove', type ='vector', name=file.split('.')[0], flags ='f')
        gs.run_command('v.extract', input =hydro_connectors, where= "crossing!='None'", output = '{}_temp'.format(hydro_connectors)) 
        gs.run_command('g.rename', vector= "{0}_temp,{0}".format(hydro_connectors))
        plot_grass_layer(hydro_connectors, str(config.log.out/"topo_carve_lines.png"), vector=True)
        gs.run_command('v.out.ogr',input=hydro_connectors,output=config.root/'outputs'/'{0}.gpkg'.format(hydro_connectors),type= 'line',format='GPKG')

def get_increases_along_streams(point_list,cat,outputs_dir,sr,ratio):
        
    #attribute geometry for analysis
    index = 1
    df_pts = pd.DataFrame(columns = ['x','y','z'])
    df_lines_out = gpd.GeoDataFrame()
    for point in point_list:
        coord = point.coords[0]
        df_pts = df_pts.append(pd.DataFrame([[coord[0],coord[1],coord[2]]],columns = ['x','y','z']))
    df_pts.reset_index(inplace=True)
    df_pts.loc[:,'delta'] = df_pts.loc[:,'z'].diff()
    df_pts.loc[:,'x_distance'] = df_pts.loc[:,'x'].diff()
    df_pts.loc[:,'y_distance'] = df_pts.loc[:,'y'].diff()
    df_pts.loc[:,'x_distance'].fillna(0.0,inplace=True)
    df_pts.loc[:,'y_distance'].fillna(0.0,inplace=True)
    df_pts['distance'] = np.sqrt(df_pts['y_distance'].apply(lambda x: x**2) + df_pts['x_distance'].apply(lambda x: x**2))
    df_pts['delta'].fillna(0.0,inplace=True)
    df_pts_out = pd.DataFrame(columns = df_pts.columns.to_list())
    increases = df_pts.loc[df_pts['delta'] > 0].index.to_list()
    seg = 1
    while increases:
        rolling_sum = df_pts.loc[increases[0]]['delta']
        i = 1
        while rolling_sum >= 0:
            rolling_sum += df_pts.loc[increases[0]+i:increases[0]+i+1]['delta'].sum()
            if rolling_sum >= 0 and increases[0]+i+1 <= df_pts.last_valid_index():
                i+= 1
                pass
            else:
                df_seg = df_pts.loc[increases[0]-1:increases[0]+i]
                road_pts = roads_detector(df_seg,ratio)
                if road_pts.empty is False:
                    road_pts = gpd.GeoDataFrame(road_pts, geometry=road_pts.apply(lambda r: shapely.geometry.Point(r.x,r.y,r.z), axis=1))
                    road_pts['segment'] = seg
                    seg+=1
                    # generate Linestrings grouping by station
                    gls = gpd.GeoDataFrame(geometry=gpd.GeoSeries(road_pts.groupby('segment').apply(lambda d: shapely.geometry.LineString(d["geometry"].values))))
                    gls.set_crs(crs = sr, inplace=True,allow_override=True)
                    df_lines_out = df_lines_out.append(gls)
                    
                remove_list = []
                for k in range(i):
                    if increases[0]+k in increases:
                        remove_list.append(increases[0]+k)
                        if increases[0]+k == df_pts.last_valid_index():
                            if increases[0]+k not in remove_list:
                                remove_list.append(increases[0]+k)
                for removal in remove_list:
                    increases.remove(removal)
                if not increases:
                    rolling_sum = -0.1 
    #add road names
    if df_lines_out.empty is False:
        nad83 = 6318
        df_lines_out = df_lines_out.to_crs(nad83)
        df_lines_out['esri_geo'] = df_lines_out['geometry'].apply(lambda x : get_esri_JSON(x.__geo_interface__,nad83))
        df_lines_out['crossing'] = df_lines_out['esri_geo'].apply(lambda x : get_road_names(x,nad83,'NAME'))
        df_lines_out = df_lines_out[['crossing','geometry']]
        df_lines_out = df_lines_out.to_crs(sr)
    #export
    if df_lines_out.empty is False:
        df_lines_out.set_crs(crs = sr, inplace=True,allow_override=True)
        df_lines_out.to_file(str(outputs_dir/'hydro_connectors_in_{}.geojson'.format(cat)),driver='GeoJSON')
        
def topo_carve(config,overwrite = False):
    dem = config.dem.name
    hydro_connectors = dem[:dem.find('dem')]+'hydro_connectors'
    reg = gs.parse_command('g.region', raster=dem, flags='pgm', zoom=dem,align=dem)
    cell_size = float(reg.nsres)
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    if "{}_hydro_enforced".format(dem) in GRASS_raster_files and overwrite is False:
        logger.info('hydro enforced dem already exists. Not recarving')
    else:
        logger.info('burning hydro_connectors into DEM')
        gs.run_command(
            'r.carve', 
            output="{}_hydro_enforced".format(dem),
            raster=dem, vector=hydro_connectors, 
            width = cell_size*2,
            depth="0.0", overwrite=True, flags="n"
        )  
    config.dem.enforced = "{}_hydro_enforced".format(dem)
    return config

def delineate_at_threshold(config,overwrite = False):
    accum = config.accum
    threshold = config.thresholds['stream']
    reg = gs.parse_command('g.region', raster=accum, flags='pgm',align=accum)
    cell_area = (float(reg.nsres)*float(reg.ewres))
    target_accum_cells = threshold / cell_area
    gs.run_command('r.mapcalc',overwrite=True,\
                      expression='above_thresh = if({0} < {1},null(),{0})'.format(accum, target_accum_cells))
    gs.run_command('r.mapcalc',overwrite=True,\
                      expression='outlet = if({0} = nmin({0}),1,null())'.format(accum))
    gs.run_command('r.to.vect',overwrite=True,\
                          input = 'outlet', output = 'outlet_vec',\
                          type='point')
    v_basin = delineate_at_point('outlet',config.drain,'thresholds')
    gs.run_command('v.out.ogr', input=  v_basin ,type = 'area',output = out_dir/'{}.gpkg'.format(v_basin), format = 'GPKG')
    
def delineate_at_point(point_v,flow_dir,basin_name):
    gs.run_command('r.stream.basins',overwrite=True,\
              direction=flow_dir,points=point_v,\
              basins = 'r_{}'.format(basin_name))
    gs.run_command('r.to.vect',input = 'r_{}'.format(basin_name),output = 'v_{}'.format(basin_name), type = 'area')
    gs.run_command('v.clean',input = 'v_{}'.format(basin_name),threshold = 5000,tool='rmarea',output = '{}_clean'.format(v_basins))
    return 'v_{}'.format(basin_name)
    
    
def patch_rasters(config, directory,out_name):
    GRASS_raster_files= [file for line in gs.list_strings(type='raster') for file in [line.split("@")[0]]]
    sr = config.sr
    tif_boundary_v = out_name+'_boundary'
    tile_rasters = []
    clean_up_rasters = [] #hold raster list for later cleanup
    li_all_files = [directory/file for file in os.listdir(directory) if os.path.splitext(file)[-1] in ['.tif','.img']]
    for raster in li_all_files:
        raster_name = ('%s' % raster.name)[:-4]
        vrt_path = project_raster(('%s' % raster.name), directory, sr)
        if raster_name not in GRASS_raster_files:
            print("adding raster {}".format(raster_name))
            gs.run_command('r.in.gdal', input=vrt_path, output= raster_name)
            gs.run_command('g.region', raster=raster_name) 

        tile_rasters.append(raster_name)

        clear_output(wait=True)

    print("Patching together {} rasters".format(len(tile_rasters)))

    if len(tile_rasters) > 1:
        gs.run_command('g.region', raster=tile_rasters)
        gs.run_command('r.patch', input=tile_rasters, output=out_name)


    else:
        gs.run_command('g.rename', raster=(tile_rasters[0],out_name))


    print('Created: '+ out_name)

    #cleanup and remove raster tiles
    for tile in tile_rasters:
        gs.run_command('g.remove', type ='raster', name=tile, flags ='f')
