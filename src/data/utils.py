import geopandas as gpd
import os, sys, copy, shutil, subprocess
from IPython.display import display, Markdown, Image
import pathlib as pl
import json
import numpy as np
import pandas as pd
import sys
import re
import urllib
import urllib.parse
import urllib.request
import requests
import io
from grass_session import Session
import grass.script as gs
import grass.script.setup as gsetup
import grass.script.array as garray
from grass.pygrass.modules import Module, ParallelModuleQueue
from grass.script import core as gcore
from matplotlib import pyplot as plt
import rasterio
import glob
from osgeo import gdal
import ssl
import json
import re
import fiona
from zipfile import ZipFile
from shapely import geometry
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter
import matplotlib.colors as colors
import datetime


import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
logger.root.setLevel(logging.INFO)
logging.basicConfig(format = '[%(asctime)s] [%(levelname)s] [%(module)s] : %(message)s')


def esri_query(parameters, level = 6):
    #current USGS wbd service
    base_url = 'https://hydro.nationalmap.gov/arcgis/rest/services/wbd/MapServer/' + str(level) + '/query?' 
    r = requests.get(base_url, parameters)
    try:
        response = json.loads(r.content)
        return(response)
    except Exception as error:
        print(f'Except raised trying to run ESRI query with params {parameters} : {error}') 
        

def get_huc12(point:tuple) -> list:
    '''
    takes a point and returns HUC12 the point is in 
    '''
    str_point = ','.join(map(str, point)) 
    parameters = {
        'outFields':'huc12',
        'geometry':str_point,
        'geometryType':'esriGeometryPoint',
        'inSR':'4269',
        'outSR':'4269',
        'spatialRel':'esriSpatialRelIntersects',
        'f':'json',
        'returnGeometry':'false'
        }
    response = esri_query(parameters)["features"]
    assert len(response) == 1, "point returned more than one HUC!"
    return response[0]["attributes"]["huc12"]

def bbox_gdf(gdf:gpd.GeoDataFrame):
    ext_d = gdf.bounds.iloc[0].to_dict()
    b_box = [str(ext_d['minx']),str(ext_d['miny']),str(ext_d['maxx']),str(ext_d['maxy'])]
    return ','.join(b_box)

def create_grass_session(huc, config,overwrite=False):
    time = datetime.datetime.now()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_out_dir = config.log.dir/("huc" + huc)/timestr
    logger.info("creating timestamped log directory: " + str(log_out_dir))
    os.makedirs(log_out_dir)

    config.log.out = log_out_dir
    huc2 = huc[:2]
    huc4 = huc[:4]
    huc8 = huc[:8]
    mapset = "ML"

    logger.info("setting up GRASS session for HUC %s, EPSG %s"%(huc, config.sr))
    location = "huc%s_sr%s" % (huc, config.sr)
    if not os.path.exists(config.gisdb):
        os.makedirs(config.gisdb, exist_ok=True)
    setup_grass_db(config.gisdb, location, config.sr, mapset,overwrite)
    if overwrite:
        gs.run_command("g.remove", type="all", pattern="", flags="f")
    logger.info("GRASS session set up: %s"%(gs.gisenv()))
    return config

    
def setup_grass_db(gisdb, location, sr, mapset,overwrite):    
    # remove prev session 
    full_location=os.path.join(gisdb, location)
    if os.path.exists(full_location) and overwrite:
        shutil.rmtree(full_location)
        
    with Session(gisdb=gisdb, location=location, create_opts="EPSG:"+sr):
        gcore.parse_command("g.gisenv", flags="s") #  run something in PERMANENT mapset
    with Session(gisdb=gisdb, location=location, mapset=mapset, create_opts=""):
        gcore.parse_command("g.gisenv", flags="s") # do something in the secondary mapset.
        
    gsetup.init(os.environ["GISBASE"], gisdb, location, mapset)

def cleanup_prev_grass_session(gisdb, location, mapset):
    folders = [pl.Path(gisdb)/location/'Permanent/.tmp/unknown', pl.Path(gisdb)/location/mapset/'.tmp/unknown']
    for folder in folders:
        if (os.path.exists(folder)):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete file %s. Reason: %s' % (file_path, e))

def mask_array(mask_array, array_to_mask):
    array_to_mask = array_to_mask.astype(float)
    mask = np.ma.masked_where(mask_array == 0, array_to_mask)
    array_to_mask[mask.mask]=np.NaN
    return array_to_mask

def get_show_string(df, n=20, truncate=True, vertical=False):
    """ returns what pyspark .show() would return, but in a string format"""
    if isinstance(truncate, bool) and truncate:
        return(df._jdf.showString(n, 20, vertical))
    else:
        return(df._jdf.showString(n, int(truncate), vertical))
    

