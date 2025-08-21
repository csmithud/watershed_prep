from affine import Affine
import argparse
import os
if os.environ["APP_ENV"].lower() == "cloud":
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
import pandas as pd
import pathlib as pl 
import rasterio
from rasterio import features
import re 
import requests
from shapely.geometry import shape, Polygon
import subprocess
import shutil
import sys
import urllib, urllib.parse, urllib.request
import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')
from zipfile import ZipFile
import fiona
from matplotlib import pyplot as plt
import rasterio

if os.getenv("AZ_BATCH_TASK_WORKING_DIR") is not None:
    sys.path.append(os.getenv("AZ_BATCH_TASK_WORKING_DIR"))

from src.config import create_config
import src.data.utils as utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  


proj_four_dict = {'102704':"+proj=lcc +lat_1=40 +lat_2=43 +lat_0=39.83333333333334 +lon_0=-100 +x_0=500000.0000000002 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs "}

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
    
def get_huc_boundary(gdb_path, out_file, huc, sr, overwrite=False):
    """
    Extracts HUC boundary from input, re-projects to desired sr, 
    and saves to given out_path.
    
    :param PosixPath huc_boundaries: full path to  file of HUC boundaries
    :param PosixPath out_dir: path to directory to place extracted/re-projected HUC boundary 
    :param str huc: HUC of interest 
    :param str sr: EPSG of desired spatial reference 
    :param logical overwrite: if the desired output files already exist, should they be recomputed?
    
    :raises AssertionError: if provided either shapefile or output directory doesn't exist
    """
    assert os.path.exists(gdb_path), "gdb_path doesn't exist: " + str(gdb_path)
       
    if overwrite == False and check_files_exist([out_file]): 
        logger.info("HUC boundary file already exists; not recalculating.")
        return 
    huc_level = len(huc)
    huc_boundaries = gpd.read_file(gdb_path,layer='WBDHU{}'.format(huc_level))
    huc_boundaries = lower_pd_cols(huc_boundaries)
    huc_boundary = huc_boundaries[huc_boundaries["huc{}".format(huc_level)] == huc]
    huc_boundary = huc_boundary.to_crs("EPSG:" + sr)
    write_file(huc_boundary, out_file)
    return huc_boundary

def lower_pd_cols(df):
    columns = df.columns.to_list()
    lower_dict = {}
    for col in columns:
        lower_dict[col] = col.lower()
    df = df.rename(columns = lower_dict)
    return df

# def tnm_query(parameters:'dict', max_iterations=5000) -> list:
#     '''
#     Uses TNM API service to get product download links 
#     returns all relevant data from rest service and must
#     be filtered by user to get desired content such as download URLs    
    
#     :param dict parameters: dictionary of parameters to pass to TNM access API (https://apps.nationalmap.gov/tnmaccess/)    
#     '''
    
#     base_url = 'https://tnmaccess.nationalmap.gov/api/v1/products?'
        
#     for i in range(max_iterations):
#         r = requests.get(base_url, parameters)
#         if r.status_code == 200:
#             response = json.loads(r.content)
#             return response 
#         else: 
#             logger.warning("USGS service error for parameter request:" + str(parameters) + "\nTrying again...")

#     logger.error("Unable to receive request for parameters :" + str(parameters))
        

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


def download_nhd(huc4:str,gdb_dir:pl.Path,overwrite=False) -> str:
    '''
    creates link to nhd zip file on USGS The National Map. Checks link before attempting 
    to download. Prints updates and returns a list of the files downloaded.
    '''
    ##update in future to get S3 bucket information and only include available files
    out_dir = gdb_dir.parent
    s3_url = 'https://prd-tnm.s3.amazonaws.com'
    product_name = 'NHDPLUS_H_'+huc4+'_HU4_GDB.zip'
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
    flow_data = flow_data[["VPUID", "geometry","FCode","GNIS_Name"]]
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
    if overwrite is False and check_files_exist([out_vrt]): 
        logger.info("files already exists; not recalculating.")
        return
    if sr in proj_four_dict.keys():
        kwargs = {'dstSRS':proj_four_dict[sr],
            'errorThreshold':0.01} 
    else:
        kwargs = {'dstSRS':'EPSG:'+sr,
            'errorThreshold':0.01}
    gdal.Warp(
        str(out_vrt), 
        str(raster_path), 
        **kwargs
    )

def process_dems(raster_paths, select_data, sr, overwrite=False):
    """
    merges all tif rasters to VRT and then reprojects to the given spatial reference.
    """
    raster_dir = pl.Path(raster_paths[0]).parent.parent
    out_vrt_big = pl.Path('{0}/dem_{1}.vrt'.format(raster_dir,select_data))
    for raster in raster_paths:
        assert os.path.exists(raster), "raster doesn't exist: " + str(raster) 
    
    if overwrite == False and check_files_exist([out_vrt]): 
        logger.info("processed DEM already exists; not overwriting.")
        return 
    
    pl.Path(raster_dir.parent).mkdir(parents=True, exist_ok=True)
    first_raster = raster_paths[0]
    if not sr:
        sr = get_raster_epsg(first_raster)
    temp_name = str(pl.Path(raster_dir).parent/str(first_raster).split("/")[-1])
    ext = temp_name[temp_name.find('.'):]
    for raster in raster_paths:
        name = str(raster)
        reproject_raster(
            raster, 
            out_vrt = name.replace(ext, ".vrt"),
            sr=sr, 
            overwrite=overwrite
        )

    reprojected_rasters = [str(raster).replace(ext, ".vrt") for raster in raster_paths]     
    merged_rasters = gdal.BuildVRT(str(out_vrt_big), reprojected_rasters)
    return out_vrt_big
        
def get_raster_epsg(file_path):
    with rasterio.open(file_path) as src:
        sr = str(src.crs.to_epsg())
    return sr

def check_dem_coverage(dem_path, huc_path,minimum_coverage = 0.95):
    """check if the DEM has full coverage of the HUC by 
    confirming the HUC doesn't intersect the DEM bounding box. 
    returns True if DEM covers full HUC, False otherwise"""
    projection = '5070' #equal area used for coverage check
    
    #get dem coverage approximation
    out_chk = dem_path.parent/'{0}_coverage.tif'.format(dem_path.name[:-4])
    with rasterio.open(dem_path) as src:
        kwds = src.profile
        warped = rasterio.warp.aligned_target(src.profile['transform'], src.profile['width'], src.profile['height'], src.profile['transform'][0]*src.profile['blockxsize'])
        kwds['driver'] = 'GTiff'
        kwds['dtype'] = rasterio.float32
        kwds['width'] = warped[1]
        kwds['height'] = warped[2]
        kwds['transform'] = warped[0]
        kwds['nodata'] = np.nan
        null_val = src.profile['nodata']
        coverage = np.array([[2]*kwds['width']]*kwds['height'],dtype=np.float32)
        for ji, window in src.block_windows(1):
            r = src.read(1, window=window)
            #slice by 10ths
            small = r[::10,::10]
            #test to see if all sliced values are null
            if np.nanmax(small) == null_val:
                coverage[ji[0],ji[1]] = np.nan
            else:
                coverage[ji[0],ji[1]] = 1
        with rasterio.open(out_chk, 'w', **kwds) as wrt:
            wrt.write(coverage,1)

    gpd_polygonized_raster = polygonize_raster(out_chk,projection)
    dem_bounds = gpd_polygonized_raster.loc[gpd_polygonized_raster['raster_val']== 1].dissolve()
    dem_bounds_normalized = dem_bounds.buffer(0)
    out_chk_vect = out_chk.parent/'{}.geojson'.format(out_chk.name[:-4])
    dem_bounds_normalized.to_file(out_chk_vect, driver="GeoJSON")
    os.unlink(out_chk)
    
    # read huc boundary 
    huc_geo = gpd.read_file(huc_path)
    huc_geo_sr = huc_geo.to_crs(dem_bounds_normalized.crs)
    huc_geo_sr.reset_index(inplace=True)

    #plot
    fix1, ax1 = plt.subplots(figsize=(30,10))

    huc_geo_sr.plot(ax = ax1, color = 'orange', edgecolor = 'white')
    dem_bounds_normalized.plot(ax=ax1, color='red',edgecolor = 'black',alpha=0.3)
    plt.show()

    #check coverage
    inter_area = dem_bounds_normalized.intersection(huc_geo_sr)
    coverage_perc = inter_area.area.iloc[0]/huc_geo_sr.area.iloc[0]
    if coverage_perc >= minimum_coverage:
        coverage = True
    else:
        coverage = False
    logger.info('coverage is %s percent' %(round(coverage_perc,3)*100))
    return coverage

def polygonize_raster(raster_path,projection):
    mask = None
    with rasterio.Env():
        with rasterio.open(raster_path) as src:
            origin_sr = src.crs
            image = src.read(1) # first band
            results = (
            {'properties': {'raster_val': v}, 'geometry': s}
            for i, (s, v) 
            in enumerate(
                features.shapes(image, mask=mask, transform=src.transform)))
           
    geoms = list(results)
    output_gdf = gpd.GeoDataFrame.from_features(geoms,crs=origin_sr)
    #return output_gdf.to_crs(epsg=projection)
    return output_gdf

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
        
def main(huc:str, config, overwrite=False):
    """
    downloads and processes the HUC boundary, NHD flow lines, DEM files and heatmaps 
    for a given HUC and spatial refernce (EPSG). 
    
    :param huc
    """    
    huc_level = len(huc)
    huc2 = huc[:2]
    huc4 = huc[:4]
    huc8 = huc[:8]
                
    # save cloud input data locally
    if os.environ["APP_ENV"].lower() == "cloud":
        credentials = DefaultAzureCredential()
        
        logger.info("downloading nlcd landcover blobs, with overwrite: " + str(overwrite)) 
        AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName={};AccountKey={};EndpointSuffix=core.windows.net".format(
            config.nlcd.storage, os.environ["GRASS_BLOB_KEY"])
        
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = AZURE_STORAGE_CONNECTION_STRING
        nlcd_blobs = list_blobs(container_name=config.nlcd.container, sub_folder_name=pl.Path(config.nlcd.in_file).parent)
        for blob in nlcd_blobs:
            logger.info(f"downloading nlcd blob {blob}")
            utils.download_blob(config.nlcd.storage, config.nlcd.container, blob, credentials, config.root, overwrite=overwrite)
              
 
    # NHD Download 
    logger.info("downloading NHD Plus GDB for HUC " + huc4)
    nhd = download_nhd(
        huc4=huc4, 
        gdb_dir=config.root/config.nhd.out_file.gdb, 
        overwrite=overwrite
    )
    local_hucs = gpd.read_file(nhd, layer='WBDHU{}'.format(huc_level))
    write_file(local_hucs,config.root/config.huc_bounds.in_file)
    
    
    # HUC boundaries 
    logger.info("reprojecting HUC {0} boundary shapefile to EPSG {1}".format(huc_level,config.sr))
    get_huc_boundary(
        gdb_path=config.root/config.nhd.out_file.gdb, 
        out_file=config.root/config.huc_bounds.out_file.geojson, 
        huc=huc, 
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
    
    # # NLCD
    # reproject_raster(config.root/config.nlcd.in_file, 
    #                  config.root/config.nlcd.out_file.vrt,
    #                  config.sr,
    #                  overwrite=overwrite)
    
    # DEMS
    huc_boundary = gpd.read_file(config.root/config.huc_bounds.out_file.geojson)
    dem_coverage = False
    res=config.res
    dems=None
    
    if os.environ["APP_ENV"].lower() == "cloud":
        logger.info("checking blob storage for DEM files")    
        dems = check_cloud_for_dem(config.dem.storage, config.dem.container, credentials, huc)
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
                str(config.root/config.huc_bounds.out_file.geojson))
            
    while dems is None or dem_coverage == False: 
        logger.info("downloading relevant DEMs")  
        dems = download_dem(
            geo_df=huc_boundary, 
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
                str(config.root/config.huc_bounds.out_file.geojson))
            
            if not dem_coverage:
                logger.info("DEMs at res {} don't cover full watershed. trying lower resolution.".format(str(res)))
                res, config = update_res(config.res, next_res(res, config), config)
        else:
            logger.info("No DEMS for resolution {}. trying lower resolution.".format(str(res)))
            res, config = update_res(config.res, next_res(res, config), config)
            logger.info("config update: " + str(config.root/config.dem.out_file.vrt))
        
     
    logger.info("all inputs processed")
    return config

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    loc = parser.add_mutually_exclusive_group(required=True)
    loc.add_argument('--point', '-pt', help="a point in lat/long coordinates", nargs=2, type=float)
    loc.add_argument('--huc', '-huc', help="a huc code", type=str)
    parser.add_argument(
        '--overwrite', 
        help="should the files redownloaded/calculated and  overwritten if they exist", 
        dest='overwrite', action='store_true'
    )
    
    args = parser.parse_args()
    
    if args.point:
        huc = utils.get_huc12(tuple(args.point))
    else:
        huc = args.huc
        
    config = create_config(huc)
    
    main(
        huc = huc, 
        config=config,
        overwrite= args.overwrite
    )
    
    
