from affine import Affine
import argparse
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
import azure
from osgeo import gdal
import geopandas as gpd
import inspect
import io 
import json
import logging
import numpy as np
from osgeo import ogr
import os
import pandas as pd
import pathlib as pl 
import rasterio
from rasterio import features
import re 
import requests
from shapely.geometry import shape
import subprocess
import shutil
import sys
import urllib, urllib.parse, urllib.request
import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
from zipfile import ZipFile
import fiona

if os.getenv("AZ_BATCH_TASK_WORKING_DIR") is not None:
    sys.path.append(os.getenv("AZ_BATCH_TASK_WORKING_DIR"))

from src.config import create_config
import src.data.utils as utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  

def check_files_exist(files): 
    return all(pl.Path(file).exists() for file in files)
    
def write_file(obj, out_file):
    if os.path.exists(pl.Path(out_file).parent) is False:
        os.makedirs(pl.Path(out_file).parent)
    if pl.Path(out_file).suffix == ".parquet":
        obj.to_parquet(out_file)
    elif pl.Path(out_file).suffix == ".geojson":
        obj.to_file(out_file, driver="GeoJSON")

def extract_path(in_path):
    out_dir = in_path.parent/'extract'
    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    memfile = in_path
    with ZipFile(memfile,'r') as openzip:
        logger.info('extracting to: '+str(out_dir))
        openzip.extractall(path=out_dir)
    files = os.listdir(out_dir)
    exts = []
    for file in files:
        if file.split('.')[-1] not in exts:
            exts.append(file.split('.')[-1])
    if 'gdb' in exts:
        assert len(files) == 1, "zipfile contained more than 1 file: " + str(files)
        return out_dir/files[0]
    elif 'shp' in exts:
        assert 'prj' in exts, "zipfile did not contain prj file"
        return out_dir
    else:
        print("zipfile did not contain gdb or shp file")
    
def get_huc12_boundary(huc12_boundaries, out_file, huc12, sr, overwrite=False):
    """
    Extracts HUC12 boundary from input, re-projects to desired sr, 
    and saves to given out_path.
    
    :param PosixPath huc_boundaries: full path to parquet file of HUC12 boundaries
    :param PosixPath out_dir: path to directory to place extracted/re-projected HUC12 boundary 
    :param str huc12: HUC12 of interest 
    :param str sr: EPSG of desired spatial reference 
    :param logical overwrite: if the desired output files already exist, should they be recomputed?
    
    :raises AssertionError: if provided either shapefile or output directory doesn't exist
    """
    assert os.path.exists(huc12_boundaries), "input file doesn't exist: " + str(huc12_boundaries)
    
    if overwrite == False and check_files_exist([out_file]): 
        logger.info("HUC12 boundary file already exists; not recalculating.")
        return 
    
    huc12_boundaries = gpd.read_parquet(huc12_boundaries)
    huc12_boundary = huc12_boundaries[huc12_boundaries["HUC12"] == huc12]
    huc12_boundary = huc12_boundary.to_crs("EPSG:" + sr)
    write_file(huc12_boundary, out_file)
    return huc12_boundary
    
def tnm_query(parameters:'dict', max_iterations=5000) -> list:
    '''
    Uses TNM API service to get product download links 
    returns all relevant data from rest service and must
    be filtered by user to get desired content such as download URLs    
    
    :param dict parameters: dictionary of parameters to pass to TNM access API (https://apps.nationalmap.gov/tnmaccess/)    
    '''
    
    base_url = 'https://tnmaccess.nationalmap.gov/api/v1/products?'
        
    for i in range(max_iterations):
        r = requests.get(base_url, parameters)
        if r.status_code == 200:
            response = json.loads(r.content)
            return response 
        else: 
            logger.warning("USGS service error for parameter request:" + str(parameters) + "\nTrying again...")

    logger.error("Unable to receive request for parameters :" + str(parameters))
        

def get_heatmap_filename(all_data_files, huc, model_type):
    relevant_files = [f for f in all_data_files if "heatmap/" in f.lower()]
    relevant_files = [f for f in relevant_files if model_type in f.lower()]
    
    # identify relevant huc 
    heatmap_file = []
    while (len(heatmap_file) == 0) and (len(huc) >= 4):
        logger.debug("trying to find heatmap file for huc: " + huc)
        heatmap_file = [f for f in relevant_files if huc == re.findall('\d+', f)[0]]
        logger.debug(heatmap_file)
        huc = huc[:-2]
    
    if len(heatmap_file) == 0:
        logger.debug("No heatmap file found")
        return None
    else:
        heatmap_file = heatmap_file[0]
        logger.debug("found heatmap file: " + heatmap_file)
        return heatmap_file

def get_heatmap_blob(storage_acct, container, credentials, model_type, huc):
    # pull in list of all blobs from container
    container_client = utils.connect_to_container(storage_acct, container, credentials)
    data_blobs = []
    for blob in container_client.list_blobs():
        data_blobs.append(blob.name)
    return get_heatmap_filename(data_blobs, huc, model_type)


def download_nhd(huc12:str,gdb_dir:pl.Path,overwrite=False) -> str:
    '''
    creates link to nhd zip file on USGS The National Map. Checks link before attempting 
    to download. Prints updates and returns a list of the files downloaded.
    '''
    ##update in future to get S3 bucket information and only include available files
    out_dir = gdb_dir.parent
    s3_url = 'https://prd-tnm.s3.amazonaws.com'
    product_name = 'NHDPLUS_H_'+huc12[:4]+'_HU4_GDB.zip'
    link = s3_url+'/StagedProducts/Hydrography/NHDPlusHR/Beta/GDB/'+product_name

    if out_dir.exists() is False:
        os.makedirs(out_dir)
    else:
        if gdb_dir.exists() is True:
            if overwrite is True:
                shutil.rmtree(gdb_dir)
            else:
                logger.info("NHD download already exists; not redownloading.")
                return gdb_dir
    logger.info('opening: '+link)
    
    r = requests.get(link, allow_redirects=True)
    assert r.status_code == 200, 'USGS service error for request'
    assert "errorMessage" not in str(r.content), 'USGS service error for request' 
    memfile = io.BytesIO(r.content)
    with ZipFile(memfile,'r') as openzip:
        logger.info('saving to: '+str(out_dir))
        openzip.extractall(path=out_dir)
    return gdb_dir

def process_flow_nodes(in_path, flow_dict, out_file, sr, overwrite=False):
    """
    Pulls out S_Node layer from FEMA GDB or shp, 
    re-projects to the given EPSG, and saves as a parquet file.
    
    :param str in_path: full path to GDB or shp
    :param str sr: EPSG to project layer to
    :param logical overwrite: if the desired output files already exist, should they be recomputed?
    """
    assert os.path.exists(in_path), "file doesn't exist: " + str(in_path)
    #if dictionary needs to be reconfigured because of leading numbers in column names
    flow_dict_rev = {}
    
    if overwrite == False and check_files_exist([out_file]): 
        logger.info("processed S_Node files already exist; not reimporting.")
        flow_keys = list(flow_dict.keys())
        
        for flow_interval in flow_keys:
            if flow_interval.lower()[0] != 'q':
                flow_dict_rev['q_'+flow_interval] = flow_dict[flow_interval]
            else:
                flow_dict_rev[flow_interval] = flow_dict[flow_interval]
        return flow_dict_rev
    #unzip if zip
    if pl.Path(in_path).suffix == ".zip":
        in_path = extract_path(in_path)
    

    if pl.Path(in_path).suffix == ".gdb":
        #handle poorly named datasets and assume this is a FEMA flow database
        avail_layers = fiona.listlayers(pl.Path(in_path))
        node_layer = [layer for layer in avail_layers if layer.lower().find('s_node') >= 0][0]
        flow_data = gpd.read_file(pl.Path(in_path), layer= node_layer)
        discharge_layer = [layer for layer in avail_layers if layer.lower().find('l_summary_discharges') >= 0][0]
        discharge_values = gpd.read_file(pl.Path(in_path), layer= discharge_layer)
        data_cols = flow_data.columns.to_list()
        flow_keys = list(flow_dict.keys())
        for flow_interval in flow_keys:
            if flow_interval not in data_cols:
                df2 = discharge_values.loc[discharge_values['EVENT_TYP']==flow_interval][['NODE_ID','DISCH']]
                df2.rename(columns={'DISCH': 'q_'+flow_interval}, inplace=True)
                flow_data = pd.merge(flow_data.copy(),df2,on='NODE_ID', how='left')
                flow_dict_rev['q_'+flow_interval] = flow_dict[flow_interval]
         #drop nodes without 1pct flow 
        for name, return_inverval in flow_dict_rev.items():
            if return_inverval == 100:
                flow_data = flow_data.loc[flow_data[name].isna() == False]
                
    else:
        flow_data = gpd.read_file(pl.Path(in_path))
    

    flow_data = flow_data.to_crs("EPSG:" + sr)
    write_file(flow_data, out_file)
    
    if flow_dict_rev:
        return flow_dict_rev
    else:
        return flow_dict
    
def process_flow_reaches(in_path, out_file, sr, overwrite=False):
    """
    Pulls out S_Hydro_Reach layer from FEMA GDB or shp, 
    re-projects to the given EPSG, and saves as a parquet file.
    
    :param str in_path: full path to GDB or shp
    :param str sr: EPSG to project layer to
    :param logical overwrite: if the desired output files already exist, should they be recomputed?
    """
    assert os.path.exists(in_path), "file doesn't exist: " + str(in_path)
    
    if overwrite == False and check_files_exist([out_file]): 
        logger.info("processed S_Hydro_Reach files already exist; not reimporting.")
        return 
    #unzip if zip
    if pl.Path(in_path).suffix == ".zip":
        in_path = extract_path(in_path)
        
    if pl.Path(in_path).suffix == ".gdb":
        #handle poorly named layers
        avail_layers = fiona.listlayers(pl.Path(in_path))
        reach_layer = [layer for layer in avail_layers if layer.lower().find('s_hydro_reach') >= 0][0]
        flow_data = gpd.read_file(in_path, layer=reach_layer)
    else:
        flow_data = gpd.read_file(in_path)
    flow_data = flow_data.to_crs("EPSG:" + sr)
    write_file(flow_data, out_file)
    
def process_nhd(gdb_path, out_file, sr, overwrite=False):
    """
    Pulls out NHDFlowline layer from path to NHD GDB, 
    re-projects to the given EPSG, and saves as a parquet file.
    
    :param str gdb_path: full path to NHD GDB
    :param str sr: EPSG to project flowline layer to
    :param logical overwrite: if the desired output files already exist, should they be recomputed?
    """
    assert os.path.exists(gdb_path), "gdb_path doesn't exist: " + str(gdb_path)
    
    if overwrite == False and check_files_exist([out_file]): 
        logger.info("processed NHD files already exists; not recalculating.")
        return 
    
    flow_data = gpd.read_file(gdb_path, layer='NHDFlowline')
    flow_data = flow_data[["VPUID", "geometry"]]
    flow_data = flow_data.to_crs("EPSG:" + sr)
    
    write_file(flow_data, out_file)

def check_cloud_for_dem(storage_acct, container, credentials, huc):
    """
    check if the DEM folder on cloud storage has a DEM available
    """
    # pull in list of all blobs from container
    container_client = utils.connect_to_container(storage_acct, container, credentials)
    data_blobs = []
    for blob in container_client.list_blobs():
        data_blobs.append(blob.name)
    
    # get files under DEM folder
    relevant_files = [f for f in data_blobs if ("dem/" in f.lower() and "tif" in f.lower() and "aux.xml" not in f.lower())]
    
    # check if file available for HUC
    dem_files = []
    while (len(dem_files) == 0) and (len(huc) >= 4):
        logger.debug("trying to find DEM file in blob storage for huc: " + huc)
        dem_files = [f for f in relevant_files if huc == re.findall('\d+', f)[0]]
        logger.debug(dem_files)
        huc = huc[:-2]
    
    if len(dem_files) == 0:
        logger.info("No DEM file found on blob storage")
        return None
    else:
        logger.info("found DEM file(s) on blob storage: " + str(dem_files))
        
        return dem_files
    
def download_dem(geo_df, dataset, out_dir:str, huc8, overwrite=False) -> list:
    '''
    Downloads DEM data for a given area (geo_df). Returns a list of the dems downloaded.
    
    :param geo_df: GeoDataFrame of area
    :param str dataset: dataset name for desired DEM resolution; one of dem_products from config
    :param str out_dir: output path to save DEMs to
    :param logical overwrite: if the DEMs have already been downloaded, should they be re-downloaded?
    '''
    pl.Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    geo_df = geo_df.to_crs("EPSG:4269") # reproject to correct SR for TNM query 
    bounds = utils.bbox_gdf(geo_df)    
    
    try:
        parameters = {
            'datasets': dataset,
            'bbox': bounds,
            'outputFormat':'JSON'
        }
        
        dem_datasets = tnm_query(parameters)
    except AssertionError:
        parameters = {
            'datasets': dataset,
            'polyCode': "huc8",
            'polyType': huc8,
            'outputFormat':'JSON'
        }
        
        dem_datasets = tnm_query(parameters)

    dem_links = []
    for product in dem_datasets['items']:
        dem_links.append(product['downloadURL'])
    logger.debug("identified downloadURLs: " + str(dem_links))
    dem_links = [link for link in dem_links if link is not None]

    tif_links = [link for link in dem_links if link.endswith(".tif")]
        
    zip_links = [link for link in dem_links if link.endswith(".zip") and bool(re.search("img", link.lower()))]
    if tif_links:
        if tif_links[0].find('current') <0: 
            if tif_links[0].find('historical') <0:
                #Fix because USGS API returns incorrect url for download for the 1/3 arc second DEM
                substr='/13/TIFF/'
                inserttxt='current/'
                tif_links = [link.replace(substr, substr + inserttxt) for link in tif_links]


    #Replace text to add in the inserted text for the fix.
    logger.debug("identified tif links: " + str(tif_links))
    logger.debug("identified zip links: " + str(zip_links))
    dems = []
    
    if len(tif_links) == 0 and len(zip_links) == 0:
        logger.info("no DEMs found for given request")
        return dems
    
    if len(tif_links) >= len(zip_links):
        # download only tif files
        for link in tif_links:
            file = link.split("/")[-1]
            output = os.path.join(out_dir, file)
            dems.append(output)
            if overwrite == False and check_files_exist([out_dir/file]):
                logger.debug("DEM link already downloaded; not re-downloading")
                continue
            r = requests.get(link, allow_redirects=True)
            assert r.status_code == 200, 'Error for DEM link: ' + str(link)
            with open (output, 'wb') as tif:
                tif.write(r.content)
    else: 
        # extract all img files from zip and convert to tif 
        for link in zip_links:
            r = requests.get(link, allow_redirects=True)
            assert r.status_code == 200, 'Error for DEM link: ' + str(link)
            memfile = io.BytesIO(r.content)
            file_name = link.split("/")[-1].split(".")[0] 
            if overwrite == False and check_files_exist([out_dir/(file_name + ".tif")]):
                dems.append(out_dir/(file_name + ".tif"))
                logger.debug("DEM already downloaded; not re-downloading")
                continue
            with ZipFile(memfile,'r') as openzip:
                gridfiles = openzip.namelist()
                img_file = [file for file in gridfiles if file[-4:] == ".img"]
                assert len(img_file) == 1, "discovered more than one img file!"
                img_file = img_file[0]
                tif_file = img_file.split('.')[0]  + ".tif"
                logger.debug("saving img {} to {} ".format(img_file, out_dir))
                f = openzip.open(img_file)
                content = f.read()
                with open(os.path.join(out_dir, img_file), 'wb') as asc:
                    asc.write(content)
                logger.debug("converting {} to tif".format(img_file))
                cmd = ["gdal_translate", "-of", "GTiff",
                       os.path.join(out_dir, img_file), 
                       os.path.join(out_dir, tif_file)]
                subprocess.call(cmd)
                dems.append(os.path.join(out_dir, tif_file))
            
    return dems

def reproject_raster(raster_path:list, out_vrt, sr:str, overwrite=False):
    '''
    Reprojects rasters to provided spatial reference and saves output as vrt file. 
    Returns a list of the paths to the reporjected rasters
    
    :param raster_paths: full paths to files to re-project
    :param str sr: EPSG spatial reference to reproject to
    :param str out_dir: output path to save rasters to
    '''
    assert os.path.exists(raster_path), "raster doesn't exist missing: " + str(raster_path)
    
    if overwrite == False and check_files_exist([out_vrt]): 
        logger.info("files already exists; not recalculating.")
        
    gdal.Warp(
        str(out_vrt), 
        str(raster_path), 
        dstSRS='EPSG:'+sr
    )

def process_dems(raster_paths, out_vrt, sr, overwrite=False):
    """
    merges all tif rasters to VRT and then reprojects to the given spatial reference.
    """
    for raster in raster_paths:
        assert os.path.exists(raster), "raster doesn't exist: " + str(raster) 
    
    if overwrite == False and check_files_exist([out_vrt]): 
        logger.info("processed DEM already exists; not overwriting.")
        return 
    
    pl.Path(out_vrt.parent).mkdir(parents=True, exist_ok=True)
    
    for raster in raster_paths:
        reproject_raster(
            raster, 
            out_vrt = str(out_vrt.parent/str(raster).split("/")[-1].replace(".tif", ".vrt")),
            sr=sr, 
            overwrite=overwrite
        )
    
    reprojected_rasters = [str(raster).split("/")[-1].replace(".tif", ".vrt") for raster in raster_paths]
    reprojected_rasters = [os.path.join(str(out_vrt.parent), file) for file in reprojected_rasters]
    merged_rasters = gdal.BuildVRT(str(out_vrt), reprojected_rasters)
    
def check_dem_coverage(dem_path, huc12_path):
    """check if the DEM has full coverage of the HUC12 by 
    confirming the HUC12 doesn't intersect the DEM bounding box. 
    returns True if DEM covers full HUC12, False otherwise"""
    # create binary array to indicate if each pixel contains data
    dem = gdal.OpenEx(dem_path)
    srcband = dem.GetRasterBand(1)
    arr = srcband.ReadAsArray()
    arr[np.isfinite(arr)] = 1
    arr[np.isinf(arr)] = 0
    
    # create polygon of DEM shape
    for vec in rasterio.features.shapes(arr, transform=Affine.from_gdal(*dem.GetGeoTransform())):
        if vec[1] == 1:
            dem_geo = shape(vec[0])
    
    # read huc12 boundary 
    shapefile = gpd.read_file(huc12_path)
    huc12_geo = shapefile.geometry.iloc[0]
    
    assert dem_geo.is_valid, "DEM geometry invalid"
    assert huc12_geo.is_valid, "HUC12 geometry invalid"
    
    return dem_geo.contains(huc12_geo)

def next_res(curr_res, config):
    curr_idx = list(config.dem_products.keys()).index(config.res)
    idx = curr_idx + 1
    res = list(config.dem_products.keys())[idx]
    return res

def update_res(curr_res, new_res, config):
    for k,v in config.dem.out_file.__dict__.items():
        curr_name = str(config.dem.out_file.__dict__[k])
        new_name = re.sub(curr_res, new_res, curr_name)
        config.dem.out_file.__dict__.update({k: new_name})
    config.res = new_res
    return new_res, config


def list_blobs(container_name:str, sub_folder_name: str):
    """
    Download the files from a given blob.
    """
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

    try:
        # instantiate a BlobServiceClient using a connection string
        blob_service_client = BlobServiceClient.from_connection_string(conn_str=connection_string)

        # point to the container with the output blobs
        container_client  = blob_service_client.get_container_client(container_name)

        # retrieve a list of the blobs
        blob_list = container_client.list_blobs(name_starts_with=sub_folder_name)
        
        # filter to those with file extension
        blob_list = [x.name for x in blob_list if pl.Path(x.name).suffix != ""]
        logger.info("found {} blobs to download: {}".format(len(blob_list), blob_list))
            
        return blob_list
    
    except Exception as ex:
        print(f'Exception: {ex}')
        
def main(huc12:str, config, overwrite=False):
    """
    downloads and processes the HUC12 boundary, NHD flow lines, DEM files and heatmaps 
    for a given HUC12 and spatial refernce (EPSG). 
    
    :param huc12
    """    
    
    huc2 = huc12[:2]
    huc4 = huc12[:4]
    huc8 = huc12[:8]
                
    # save cloud input data locally
    if os.environ["APP_ENV"].lower() == "cloud":
        credentials = DefaultAzureCredential()
        
        riverine_flow_nodes = config.riverine_flow_nodes.in_file.as_posix() #add in function fo find geographically
        riverine_flow_reaches = config.riverine_flow_reaches.in_file.as_posix() #add in function fo find geographically
        if riverine_flow_nodes != 'None':
            logger.info("downloading riverine node blob, with overwrite: " + str(overwrite))
            utils.download_blob(config.riverine_flow_nodes.storage, config.riverine_flow_nodes.container, 
                            config.riverine_flow_nodes.in_file, credentials, config.root, overwrite=overwrite)
            if config.FEMA_map_type.find('.geojson')>=0:
                logger.info("downloading riverine floodplains from blob, with overwrite: " + str(overwrite))
                utils.download_blob(config.riverine_flow_nodes.storage, config.riverine_flow_nodes.container, 
                                config.FEMA_map_type, credentials, config.root, overwrite=overwrite)
            logger.info("downloading riverine reaches blob, with overwrite: " + str(overwrite))
            if riverine_flow_reaches != 'None':
                if config.riverine_flow_reaches.in_file != config.riverine_flow_nodes.in_file:
                    utils.download_blob(config.riverine_flow_reaches.storage, config.riverine_flow_reaches.container, 
                                config.riverine_flow_reaches.in_file, credentials, config.root, overwrite=overwrite)
        
        
        logger.info("downloading nlcd landcover blobs, with overwrite: " + str(overwrite)) 
        AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName={};AccountKey={};EndpointSuffix=core.windows.net".format(
            config.nlcd.storage, os.environ["GRASS_BLOB_KEY"])
        
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = AZURE_STORAGE_CONNECTION_STRING
        nlcd_blobs = list_blobs(container_name=config.nlcd.container, sub_folder_name=pl.Path(config.nlcd.in_file).parent)
        for blob in nlcd_blobs:
            logger.info(f"downloading nlcd blob {blob}")
            utils.download_blob(config.nlcd.storage, config.nlcd.container, blob, credentials, config.root, overwrite=overwrite)
        
        logger.info("identifying relevant heatmap")
        heatmap_file = get_heatmap_blob(config.heatmap.storage, config.heatmap.container, credentials, config.model_type, huc12)
        if heatmap_file is not None: 
            logger.info("downloading heatmap: " + heatmap_file)
            utils.download_blob(config.heatmap.storage, config.heatmap.container, 
                                heatmap_file, credentials, config.root, overwrite=overwrite)            
            
    elif os.environ["APP_ENV"].lower() == "local":
        logger.info("identifying relevant heatmap file")
        all_data_files = [str(f) for f in list(pl.Path(config.root).rglob("*.tif"))]
        logger.debug("searching data files: " + str(all_data_files))
        heatmap_file = get_heatmap_filename(all_data_files, huc12, config.model_type)
        logger.debug("heatmap file: "  + str(heatmap_file))

        
    # NHD Download 
    logger.info("downloading NHD Plus GDB for HUC " + huc4)
    nhd = download_nhd(
        huc12=huc12, 
        gdb_dir=config.root/config.nhd.out_file.gdb, 
        overwrite=overwrite
    )
    local_huc12s = gpd.read_file(nhd, layer='WBDHU12')
    write_file(local_huc12s,config.root/config.huc12_bounds.in_file)
    
    
    # HUC12 boundaries 
    logger.info("reprojecting HUC12 boundary shapefile to EPSG {}".format(config.sr))
    get_huc12_boundary(
        huc12_boundaries=config.root/config.huc12_bounds.in_file, 
        out_file=config.root/config.huc12_bounds.out_file.geojson, 
        huc12=huc12, 
        sr=config.sr, 
        overwrite=overwrite
    )
    

    #NHD Flowlines
    logger.info("extracting NHDFlowline layer and reprojecting")
    process_nhd(
        gdb_path=config.root/config.nhd.out_file.gdb,
        out_file=config.root/config.nhd.out_file.geojson,
        sr=config.sr, 
        overwrite=overwrite
    )
    
    #FEMA Flow nodes
    riverine_flow_nodes = config.riverine_flow_nodes.in_file.as_posix()
    if riverine_flow_nodes != 'None':
        logger.info("extracting FEMA Flow Nodes layer and reprojecting")
        config.name_events = process_flow_nodes(
            in_path=config.root/config.riverine_flow_nodes.in_file,
            flow_dict = config.name_events,
            out_file=config.root/config.riverine_flow_nodes.out_file.geojson,
            sr=config.sr, 
            overwrite=overwrite
        ) 
        
    #FEMA Flow reaches
    riverine_flow_reaches = config.riverine_flow_reaches.in_file.as_posix()
    if riverine_flow_reaches != 'None':
        logger.info("extracting FEMA Flow Reach layer and reprojecting")
        process_flow_reaches(
            in_path=config.root/config.riverine_flow_reaches.in_file,
            out_file=config.root/config.riverine_flow_reaches.out_file.geojson,
            sr=config.sr, 
            overwrite=overwrite
        )     
    
    # NLCD
    reproject_raster(config.root/config.nlcd.in_file, 
                     config.root/config.nlcd.out_file.vrt,
                     config.sr,
                     overwrite=overwrite)
    
    # DEMS
    huc12_boundary = gpd.read_file(config.root/config.huc12_bounds.out_file.geojson)
    dem_coverage = False
    res=config.res
    dems=None
    
    if os.environ["APP_ENV"].lower() == "cloud":
        logger.info("checking blob storage for DEM files")    
        dems = check_cloud_for_dem(config.dem.storage, config.dem.container, credentials, huc12)
        if dems is not None: 
            logger.info("checking resolution of DEM files found ")
            new_res = None
            for r in config.dem_products.keys():
                new_dems = [f for f in dems if r in f]
                if len(new_dems) > 0:
                    new_res = r
                    break
            assert new_res is not None, "Unable to find resolution name in DEMs from cloud storage"
            dems = new_dems
            logger.info("downloading DEM files: " + str(dems))
            for f in dems:
                utils.download_blob(config.dem.storage, config.dem.container, 
                                    f, credentials, config.root/"raw_dem", overwrite=overwrite)
            logger.warn("updating config resolution to " + new_res)  
            res, config = update_res(config.res, new_res, config)
            logger.info("updated res: " + config.res)
            dems = [config.root/"raw_dem"/dem for dem in dems]
            
            logger.info("reprojecting and merging DEMs")
            process_dems(
                raster_paths=dems, 
                out_vrt=config.root/config.dem.out_file.vrt, 
                sr=config.sr, 
                overwrite=overwrite
            )  
            
            logger.info("checking DEM coverage for {}".format(str(config.root/config.dem.out_file.vrt)))
            dem_coverage = check_dem_coverage(
                str(config.root/config.dem.out_file.vrt), 
                str(config.root/config.huc12_bounds.out_file.geojson))
            
    while dems is None or dem_coverage == False: 
        logger.info("downloading relevant DEMs")  
        dems = download_dem(
            geo_df=huc12_boundary, 
            dataset=config.dem_products[res], 
            out_dir=config.root/"raw_dem", 
            huc8=huc8,
            overwrite=overwrite
        )  
 
        if len(dems) > 0:
            logger.info("reprojecting and merging DEMs")
            process_dems(
                raster_paths=dems, 
                out_vrt=config.root/config.dem.out_file.vrt, 
                sr=config.sr, 
                overwrite=overwrite
            )  
            
            logger.info("checking DEM coverage for {}".format(str(config.root/config.dem.out_file.vrt)))
            dem_coverage = check_dem_coverage(
                str(config.root/config.dem.out_file.vrt), 
                str(config.root/config.huc12_bounds.out_file.geojson))
            
            if not dem_coverage:
                logger.info("DEMs at res {} don't cover full watershed. trying lower resolution.".format(str(res)))
                res, config = update_res(config.res, next_res(res, config), config)
        else:
            logger.info("No DEMS for resolution {}. trying lower resolution.".format(str(res)))
            res, config = update_res(config.res, next_res(res, config), config)
            logger.info("config update: " + str(config.root/config.dem.out_file.vrt))
        
    # HEATMAPS
    if heatmap_file is not None: 
        logger.info("reprojecting heatmap")
        reproject_raster(
            raster_path=config.root/heatmap_file, 
            out_vrt=config.root/config.heatmap.out_file.vrt, 
            sr=config.sr, 
            overwrite=overwrite
        )
     
    logger.info("all inputs processed")
    return config

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    loc = parser.add_mutually_exclusive_group(required=True)
    loc.add_argument('--point', '-pt', help="a point in lat/long coordinates", nargs=2, type=float)
    loc.add_argument('--huc12', '-huc', help="a huc12 code", type=str)
    parser.add_argument(
        '--overwrite', 
        help="should the files redownloaded/calculated and  overwritten if they exist", 
        dest='overwrite', action='store_true'
    )
    
    args = parser.parse_args()
    
    if args.point:
        huc12 = utils.get_huc12(tuple(args.point))
    else:
        huc12 = args.huc12
        
    config = create_config(huc12)
    
    main(
        huc12 = huc12, 
        config=config,
        overwrite= args.overwrite
    )
    
    
