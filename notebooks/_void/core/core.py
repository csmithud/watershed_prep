import numpy as np
from matplotlib import pyplot as plt
from pysheds.grid import Grid
import folium
import pathlib as pl
import os
import geopandas as gpd
import rasterio
import pandas as pd
import gdal
import pyproj
from scipy.ndimage import gaussian_filter
from affine import Affine
from shapely import geometry
from skimage.morphology import skeletonize
from scipy import ndimage

import matplotlib.colors as colors

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

#Bounds based on geometry of geodateframe limits
def bounds_lat_lon(geo_df: gpd.GeoDataFrame, Buffer: float, Coord_System: int):
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