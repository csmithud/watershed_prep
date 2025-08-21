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
if os.environ["APP_ENV"].lower() == "cloud":
    from azure.storage.blob import BlobServiceClient, generate_account_sas, ResourceTypes, AccountSasPermissions
    from azure.identity import DefaultAzureCredential
    from azure.identity import DeviceCodeCredential
    from azure.keyvault.secrets import SecretClient 
    from pyspark.sql import SparkSession
    import pyspark
    from delta import *
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
    if os.environ["APP_ENV"].lower() == "local":
        time = datetime.datetime.now()
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_out_dir = config.log.dir/("huc" + huc)/timestr
        logger.info("creating timestamped log directory: " + str(log_out_dir))
        os.makedirs(log_out_dir)

    if os.environ["APP_ENV"].lower() == "cloud":
        logger.info("creating temporary log directory")
        log_out_dir = pl.Path("log")
        os.makedirs(log_out_dir)
        logger.info("setting up credentials to write to blob storage")
        credentials = DefaultAzureCredential()
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

def download_blob(storage, container, blob, credentials, tmp_dir, overwrite = True):
    if (tmp_dir/blob).exists() and overwrite == False:
        print("blob already downloaded")
        return 
    
    pl.Path((tmp_dir/blob).parent).mkdir(parents=True, exist_ok=True)
    
    blob_service_client = BlobServiceClient(account_url="https://{}.blob.core.windows.net/".format(storage), credential=credentials)    
    blob_client = blob_service_client.get_blob_client(container=str(container), blob=str(blob))
    
    assert blob_client.exists(), "blob doesn't exist!"

    with open(str(tmp_dir/blob), "wb") as download_file:
        download_file.write(blob_client.download_blob().readall())
    
def connect_to_blob(storage_acct, container, blob, credentials):
    blob_service_client = BlobServiceClient(account_url="https://{}.blob.core.windows.net/".format(storage_acct), credential=credentials)    
    blob_client = blob_service_client.get_blob_client(container=str(container), blob=str(blob))
    return blob_client

def connect_to_container(storage_acct, container, credentials):
    blob_service_client = BlobServiceClient(account_url="https://{}.blob.core.windows.net/".format(storage_acct), credential=credentials)    
    container_client = blob_service_client.get_container_client(container=str(container))
    return container_client

def upload_blob(upload_file_path, storage, container, blob, credentials, overwrite = True):
    blob_client = connect_to_blob(storage, container, blob, credentials)
    
    with open(upload_file_path, "rb") as data:
        blob_client.upload_blob(data, overwrite=overwrite)

def get_show_string(df, n=20, truncate=True, vertical=False):
    """ returns what pyspark .show() would return, but in a string format"""
    if isinstance(truncate, bool) and truncate:
        return(df._jdf.showString(n, 20, vertical))
    else:
        return(df._jdf.showString(n, int(truncate), vertical))
    
def get_spark_storage(storage_account_name, application_id, password, tenant_id):
    """
    Gets a spark session with a connection to an Azure storage account 
    and the ability to write to Delta. 
    If the session has already been created, it will be returned.

    Resources: # https://docs.delta.io/latest/quick-start.html https://docs.delta.io/latest/delta-storage.html
    """
    # use storage account name as session name
    builder = (SparkSession.builder.appName(storage_account_name)
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.driver.memory", "22g")
    .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")
    .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true") 
    .config("spark.delta.logStore.class", "org.apache.spark.sql.delta.storage.AzureLogStore"))
    
    spark = spark = configure_spark_with_delta_pip(builder) \
    .config("spark.jars.packages", ",".join(["io.delta:delta-core_2.12:1.1.0", "org.apache.hadoop:hadoop-azure:3.3.1", "org.apache.hadoop:hadoop-azure-datalake:3.3.1"])) \
    .getOrCreate()     
    
    spark.conf.set("fs.azure.account.auth.type.{}.dfs.core.windows.net".format(storage_account_name), "OAuth")
    spark.conf.set("fs.azure.account.oauth.provider.type.{}.dfs.core.windows.net".format(storage_account_name),  "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
    spark.conf.set("fs.azure.account.oauth2.client.id.{}.dfs.core.windows.net".format(storage_account_name), application_id)
    spark.conf.set("fs.azure.account.oauth2.client.secret.{}.dfs.core.windows.net".format(storage_account_name), password)
    spark.conf.set("fs.azure.account.oauth2.client.endpoint.{}.dfs.core.windows.net".format(storage_account_name), "https://login.microsoftonline.com/{}/oauth2/token".format(tenant_id))
    
    return spark
