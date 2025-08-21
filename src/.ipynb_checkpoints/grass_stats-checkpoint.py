import argparse
from azure.identity import DefaultAzureCredential
import copy
from functools import reduce
from osgeo import gdal
import geopandas as gpd
from grass_session import Session
import grass.script as gs
import grass.script.setup as gsetup
import grass.script.array as garray
from grass.pygrass.modules import Module, ParallelModuleQueue
from grass.script import core as gcore
import inspect
from itertools import product
import logging 
import math
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import multiprocessing as multi
import numpy as np
import os
import pandas as pd
import pathlib as pl
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import *
import rasterio
import re 
import sys
import time
import requests
import json
from shapely.geometry import shape
import warnings; warnings.filterwarnings('ignore', message='.*initial implementation of Parquet.*')

#from delta.tables import * 

if os.getenv("AZ_BATCH_TASK_WORKING_DIR") is not None:
    sys.path.append(os.getenv("AZ_BATCH_TASK_WORKING_DIR"))

from src.config import create_config
import src.data.utils as utils

# start logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  
    
 

def igen(a, n, m):
    """
    generator for splitting up 2d array into smaller chunks 
    reference: https://stackoverflow.com/questions/41214432/how-do-i-split-a-2d-array-into-smaller-2d-arrays-of-variable-size
    """
    i_ = np.arange(a.shape[0]) // n
    j_ = np.arange(a.shape[1]) // m
    for i, j in product(np.unique(i_), np.unique(j_)):
        yield (i, j), a[i_ == i][:, j_ == j]
        
def layers_to_spark_df(huc12, layers, mask, gs, spark, tile_size=256):
    
    # initialize empty dataframe 
    df = pd.DataFrame(data=[], index=None)
    
    # loop through layers, create chunks for each, and combine to DF
    for i,layer in enumerate(layers):
        logger.info("adding layer " + layer)
        
        # create chunks
        array = utils.mask_array(mask, garray.array(layer))
        array = np.array(array, np.float32)
        if 'heatmap' not in layer:
            array = np.round(array, 2)
        dict_of_arrays = dict(igen(array, tile_size, tile_size))   
   
        # flatten arrays for each chunk
        flat_array = [i.flatten() for i in list(dict_of_arrays.values())]

        if i == 0:
            logger.info("pulling metadata from first layer")
            # create indices for each chunk
            array_idx = [np.array(i, dtype=np.int32) for i in list(dict_of_arrays.keys())]
            array_idx_x = [x[0] for x in array_idx]
            array_idx_y = [x[1] for x in array_idx]
            n_tiles = len(array_idx)
        
            nrows, ncols = np.shape(array)
            gs.run_command('r.out.gdal', input=layer, output="tmp.vrt", format="VRT")
            with rasterio.open("tmp.vrt") as r_dataset:
                xmin, ymin, xmax, ymax = r_dataset.bounds
                xres = (xmax-xmin)/float(ncols)
                yres = (ymax-ymin)/float(nrows)
                crs = str(r_dataset.meta["crs"])
            os.remove("tmp.vrt")  
    
            # create lists to hold each chunk's dimensions and lat/long coverage
            lat_list=[]
            long_list=[]
            dim_list=[]
            for key, value in dict_of_arrays.items():
                rows_a,cols_a=value.shape
                dim_list.append(np.array([rows_a,cols_a], dtype=np.int32))
                lat_list.append(np.array([ymin+yres*tile_size*key[0], ymin+yres*(tile_size*key[0]+rows_a)],dtype=np.float32))
                long_list.append(np.array([xmin+xres*tile_size*key[1], xmin+xres*(tile_size*key[1]+cols_a)],dtype=np.float32))
                
            df.insert(0, "huc", [huc12]*n_tiles)
            df.insert(len(df.columns), "idx_0", array_idx_x)
            df.insert(len(df.columns), "idx_1", array_idx_y)
            df.insert(len(df.columns), "dims", dim_list)
            df.insert(len(df.columns), "crs", [crs]*n_tiles)
            df.insert(len(df.columns), "lat", lat_list)
            df.insert(len(df.columns), "long", long_list)

        # insert layer
        df.insert(len(df.columns), layer, flat_array)
       
    logger.info("turning into spark dataframe")
    sdf = spark.createDataFrame(df)
    
    return sdf 

def bound_box_dict(nsew:list,increment:int):
    features = []
    iterate = range(increment)
    bounds = {'xmin':float(nsew[3]),'ymin':float(nsew[1]),'xmax':float(nsew[2]),'ymax':float(nsew[0])}
    startY_big = bounds['ymin']
    startX_big = bounds['xmin']
    diffY = (bounds['ymax'] - startY_big)/increment
    diffX =  (bounds['xmax'] - startX_big)/increment
    for x,y in [(x,y) for x in iterate for y in iterate]:
        startY = startY_big + diffY*(y)
        startX = startX_big + diffX*(x)
        endY = startY_big + diffY*(y+1)
        endX = startX_big + diffX*(x+1)
        b_box = '{'+"'xmin':'{0}','ymin':'{1}','xmax':'{2}','ymax':'{3}'".format(str(startX),str(startY),str(endX),str(endY))+'}'
        features.append(b_box)
    return features

def layer_indexer(url,params,layer):
    r = requests.get(url,params)
    rest_layers = json.loads(r.content)['layers']
    for l in rest_layers:
        if l['name'] == layer:
            return l['id']

def get_FEMA_floodplains_as_geojson(out_dir:pl.Path ,grass_bounds_layer:str,sr,flow_keys,map_type, config):
    #empty dict of geojson folders to be returned
    geojsons = {}
    #create output folder
    if not os.path.exists(out_dir/'floodplains'): 
        os.makedirs(out_dir/'floodplains')
    floodplain_dir = out_dir/'floodplains'
    #get bounds of huc12 + buffer
    nsew = list(gs.parse_command('v.to.db',map=grass_bounds_layer,columns='bbox',flags='p',option='bbox').keys())[1].split('|')[1:]
    #get url to flood hazard zones (built to handle prelim, effective and pending). Also built to handle layer changes on FEMA's end
    layer = {'effective':'Flood Hazard Zones','preliminary':'Preliminary Flood Hazard Zones','pending':'Pending Flood Hazard Zones'}
    map_types ={'effective':'public/NFHL/MapServer/','preliminary': 'PrelimPending/Prelim_NFHL/MapServer/', 'pending': 'PrelimPending/Pending_NFHL/MapServer/'}
    base_url = 'https://hazards.fema.gov/gis/nfhl/rest/services/'+map_types[map_type]
    params = {'f':'json'}
    layer_index = layer_indexer(base_url,params,layer[map_type])
    query_url = '{0}/{1}/query?'.format(base_url,layer_index)
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.61 Safari/537.36'}
    for flow_key in flow_keys:
        flow_interval = config.name_events[flow_key]
        logger.info("getting {0} floodplains for {1} flow interval".format(map_type,flow_interval))
        if flow_interval == 100:
            whereC = "SFHA_TF = 'T' OR SFHA_TF = 'F' AND ZONE_SUBTY = '1 PCT DEPTH LESS THAN 1 FOOT'"
        elif flow_interval == 500:
            whereC = "SFHA_TF = 'T' OR (SFHA_TF = 'F' AND ZONE_SUBTY IN ('0.2 PCT ANNUAL CHANCE FLOOD HAZARD', 'AREA WITH REDUCED FLOOD RISK DUE TO LEVEE'))"
        else:
            #placeholder for when we have floodplains for more recurrence intervals
            whereC = None
        assert whereC, "Function is only designed to handle 100 and 500 year intervals at this time"
        tiles = 0
        tiles_needed = 1
        while tiles != tiles_needed:
            tiles = tiles_needed
            bounds = bound_box_dict(nsew,tiles)
            for tile in bounds:
                query_params = {'where': whereC,'f':'geojson','geometry':tile,'returnGeometry':'true','geometryType':'esriGeometryEnvelope','inSR':sr,'outSR':sr}
                response = requests.get(query_url,query_params, headers = headers)
                json_response = json.loads(response.content)
                if 'exceededTransferLimit' in json_response.keys():
                    tiles_needed = tiles*2
                    tiles = 0
                    break
                else:
                    if 'features' in json_response.keys():
                        df = pd.DataFrame(json_response['features'])
                        if 'geometry' in df.columns:
                            df['geometry'] = df['geometry'].apply(lambda x: shape(x))
                            gdf = gpd.GeoDataFrame(df, geometry = df['geometry'],crs = 'EPSG:{}'.format(sr))
                            index = bounds.index(tile)
                            if index == 0:
                                gdf_compile = gdf
                            else:
                                gdf_compile = gdf_compile.append(gdf)

        gdf_compile.to_file(floodplain_dir/'FEMA_{0}_{1}.geojson'.format(map_type,flow_interval),driver="GeoJSON")
        geojsons[flow_interval] = floodplain_dir/'FEMA_{0}_{1}.geojson'.format(map_type,flow_interval)
    return geojsons
                
def create_FEMA_AEP_grids(grass_map,config):
    flow_keys = list(config.name_events.keys()) 
    sr = config.sr
    map_type = config.FEMA_map_type
    out_dir = pl.Path(config.root)
    #get FEMA flood hazard layer for each event
    if config.FEMA_map_type.find('.geojson')>=0:
        map_type =config.FEMA_map_type
        floodplain = gpd.read_file(config.root/map_type)
        floodplain = floodplain.to_crs("EPSG:" + sr)
        out_file = config.root/map_type.replace('.geojson','_processed.geojson')
        floodplain.to_file(out_file, driver="GeoJSON")
        geojsons_foodplain = {100:out_file}
    else:
        geojsons_foodplain = get_FEMA_floodplains_as_geojson(out_dir= out_dir,grass_bounds_layer= grass_map, sr=sr, flow_keys=flow_keys, map_type = map_type,config = config)
    riverine_floodplain_rasters = []
    for event in geojsons_foodplain.keys():
        gs.run_command(
            'v.import', 
            input= geojsons_foodplain[event], 
            output="floodplain_{}".format(event), 
            overwrite = True
        )
        gs.run_command(
            'v.to.rast', 
            input= "floodplain_{}".format(event), 
            type='area',
            use='val',
            value= float(1/event),
            output="floodplain_{}".format(event), 
            overwrite = True
        )
        riverine_floodplain_rasters.append("floodplain_{}".format(event))
    aep_grid_name = 'heatmap_riverine'
    gs.run_command(
                'r.mapcalc',
                expression= "{0} =  nmax({1})".format(aep_grid_name,','.join(riverine_floodplain_rasters))
            )
    return aep_grid_name

def create_flow_grids(flow_points:str,flow_keys:str,drain_dir:str,config):
    '''function to convert the FEMA flow grids into raster stream alignments with the flow attributed
    to the respective stream
    '''
    node_dict = {}
    flow_maps = {}
    max_flow_maps = []
    map_type = []
    node_out = list(gs.parse_command('v.to.db',map=flow_points,columns='x_y',flags='p',option='coor').keys())

    for flow_key in flow_keys:
        flow_interval = config.name_events[flow_key]
        flow_maps[flow_interval] = []
    
    for node in node_out[1:]:
        
        #getting flow node drain path
        #r.drain test huc12 took 34min to process 
        #upgrade later to split work 
        #     workers = multi.cpu_count()
        #     rdrain = Module("r.drain", overwrite=True, run_=False)
        variables = node.split('|')
        gs.run_command(
            'r.mapcalc',
            expression= "drain_deg = if({0} != 0, 45. * abs({0}), null())".format(drain_dir)
        )
        gs.run_command(
            'r.mapcalc',
            expression = "const1 = 1"
        )
        gs.run_command(
            'r.drain', 
            input = 'const1',
            direction = 'drain_deg',
            output="f_drain_{0}".format(variables[0]),
            start_coordinates=','.join([variables[1],variables[2]]),
            overwrite = True
        )

        #multiplying the node direction by flow magnitude
        for flow_key in flow_keys:
            flow_value = gs.parse_command('v.db.select',map=flow_points,where="{0} = '{1}'".format('cat',variables[0]),columns = flow_key)
            #only add nodes where flow values exists
            if flow_value != '':
                flow_interval = config.name_events[flow_key]
                gs.run_command(
                'r.mapcalc', 
                expression="f_drain_{0}_{1}_flow =  f_drain_{1} * {2}".format(flow_interval,variables[0],list(flow_value.keys())[1])
                )
                flow_maps[flow_interval].append('f_drain_{0}_{1}_flow'.format(flow_interval,variables[0])) 
                map_type.append('raster')
        
        ##removed for speed
        #gs.run_command('g.remove', type='raster', name="f_drain_{0}".format(variables[0]), flags='f')
    
    #get max flow values from each drained node
    for flow_key in flow_keys:
        flow_interval = config.name_events[flow_key]
        gs.run_command(
        'r.mapcalc', 
        expression="f_drain_flow_{0} =  nmax({1})".format(flow_interval,','.join(flow_maps[flow_interval]))
        )
        gs.run_command(
        'g.remove',
        type=','.join(map_type),
        name=','.join(flow_maps[flow_interval]),
        flags='f'
        )
        max_flow_maps.append('f_drain_flow_{0}'.format(flow_interval))
    
    return max_flow_maps
    
    
def run_parallel(mod, workers, jobs, **kwargs):
    """
    Runs GRASS modules in parallel. All arguments for the given mod should be supplied
     in the kwargs.
    
    :param mod: GRASS Module object 
    :param int workers: number of available CPUs for parallel processing
    :param list jobs: list of unique names for jobs to run. 
    :param **kwargs: additional options to pass to the module. If any option should include
        the job name (e.g. to name a new layer) supply a lambda function. For example, for a
        mapcalc, the kwargs could be:
        expression = lambda job: '%s = %s * 5'%(layer[1] + _ job, layer[0])
    """
    queue = ParallelModuleQueue(nprocs=workers)
    mod_list = []
            
    for job in jobs:
        new_mod = copy.deepcopy(mod)
        mod_list.append(new_mod)
        
        # fill in any lambda functions with job name
        new_kwargs =  copy.deepcopy(kwargs)
        for k,i in kwargs.items():
            if callable(i):
                new_kwargs[k] = new_kwargs[k](job)   
        m = new_mod(**new_kwargs)   
        queue.put(m)
        
    queue.wait()
    mod_list = queue.get_finished_modules()
    for mod in mod_list:
         assert mod.popen.returncode == 0, "error running module in parallel" 
    return mod_list

def plot_grass_layer(layer, file_path, vector=False):
    """ 
    Plot a specific grass layer to a file. 
    Note the watershed boundary + buffered boundary should be defined as 
    hu12 and huc12buff respectively
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
        gs.run_command('d.vect', map="huc12", type="boundary", color="black", width="3")
        gs.run_command('d.vect', map="huc12buff", type="boundary", color="black", width="1")
    except:
        logger.warning(f"Couldn't find watershed boundary and/or buffered watershed boundary!")
    
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

def main(huc12, config):
    """
    Imports DEM, NHD flowline, and heatmap for given huc12 and calculates
    data for ML. Logs are saved to a folder within the log location 
    specified in the config. 
    """
    if os.environ["APP_ENV"].lower() == "local":
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_out_dir = config.log.dir/("huc" + huc12)/timestr
        logger.info("creating timestamped log directory: " + str(log_out_dir))
        os.makedirs(log_out_dir)

    if os.environ["APP_ENV"].lower() == "cloud":
        logger.info("creating temporary log directory")
        log_out_dir = pl.Path("log")
        os.makedirs(log_out_dir)
        logger.info("setting up credentials to write to blob storage")
        credentials = DefaultAzureCredential()
    
    huc2 = huc12[:2]
    huc4 = huc12[:4]
    huc8 = huc12[:8]
    mapset = "ML"
    jobs = list(config.thresholds.stream.keys())
    
    logger.info("setting up GRASS session for HUC %s, EPSG %s"%(huc12, config.sr))
    location = "huc%s_sr%s" % (huc12, config.sr)
    if not os.path.exists(config.gisdb):
        os.makedirs(config.gisdb, exist_ok=True)
    utils.setup_grass_db(config.gisdb, location, config.sr, mapset)
    gs.run_command("g.remove", type="all", pattern="", flags="f")
    logger.info("GRASS session set up: %s"%(gs.gisenv()))

    logger.info("importing huc12 boundary")
    gs.run_command(
        'v.import', 
        input=config.root/config.huc12_bounds.out_file.geojson, 
        output="huc12", 
        overwrite = True
    ) 
    
    logger.info("setting up buffer around boundary")
    
    gs.run_command(
        'v.buffer', 
        input="huc12", 
        output="huc12buff",
        distance=config.huc12_buff # distance in meters
    ) 
    #create buffer within full buffer
    gs.run_command(
    'v.buffer', 
    input="huc12", 
    output="huc12buff_95pct",
    distance=config.huc12_buff*0.95 # distance in meters
    ) 
        

    logger.info("reading DEM file")
    gs.run_command(
        'r.import', 
        input=config.root/config.dem.out_file.vrt, 
        output="dem_raw", 
        resample="bilinear",
        resolution="value",
        resolution_value= int(config.res[:-1]),
        overwrite=True, memory = config.mem
    )
    
    logger.info("temporarily setting region to full DEM")
    reg = gs.parse_command('g.region', raster="dem_raw", res=int(config.res[:-1]), flags='pgm', zoom="dem_raw",align="dem_raw")
    logger.info(reg)
    cell_size = float(reg.nsres)
    logger.info(cell_size)

    gs.run_command('r.fill.stats', input = 'dem_raw',output = 'dem', distance=5,mode='wmean',power=2,cells=4,flags='k')
    gs.run_command(
        'g.remove', 
        type='raster',
        name = 'dem_raw',
        flags='f'
    )


    
    
    logger.info("setting mask to buffered huc12")
    gs.run_command('r.mask', vector="huc12buff")
    logger.info("clipping dem to buffered huc12 and setting region")
    gs.mapcalc("dem_clipped=dem")
    reg = gs.parse_command('g.region', raster="dem_clipped", res=int(config.res[:-1]), flags='pgm', zoom="dem_clipped",align="dem_clipped")
    logger.info(reg)
    logger.info("plotting DEM and HUC boundary")
    plot_grass_layer("dem_clipped", str(log_out_dir/"dem.png"))

    # import remaining inputs
    #import pdb; pdb.set_trace()
    logger.info("importing NLCD land cover")
    gs.run_command(
        'r.import', 
        input=config.root/config.nlcd.out_file.vrt, 
        output="landcover", 
        extent="region", overwrite = True
    ) 
    
    # calculate mannning's n from land cover
    # ref: https://grasswiki.osgeo.org/wiki/NLCD_Land_Cover
    logger.info("reclassing NLCD to manning's n")
    gs.run_command(
        'r.reclass', 
        input="landcover", 
        output="mannings_n_int", 
        rules = os.path.join(str(pl.Path(__file__).parent), "nlcd_mannings.txt"),          
        overwrite = True
    )
    
    gs.mapcalc("mannings_n = mannings_n_int/10000.0")
    plot_grass_layer("mannings_n", str(log_out_dir/"mannings_n.png"))

    logger.info("importing NHD")
    gs.run_command(
        'v.import', 
        input=config.root/config.nhd.out_file.geojson, 
        output="tmp_flowline", 
        extent="region", overwrite = True
    ) 
    #clipping at 95% to avoid pulling in nulls
    gs.run_command(
    'v.clip', 
    input="tmp_flowline",
    clip='huc12buff_95pct',
    output="flowline", 
    overwrite = True
    )
    gs.run_command(
    'g.remove', 
    type='vector',
    name = 'tmp_flowline',
    flags='f'
    )
       
    
    # burn known streams into the DEM for pluvial calculations
    gs.run_command(
        'r.carve', 
        output="burneddem",
        raster="dem", vector="flowline", 
        depth="10.0", overwrite=True, flags="n"
    ) 
    logger.info('burned streams into DEM')
        
    # calculate watershed properties (accumulation, drainage direction, basins)
    # for the burned DEM 
    # Threshold is somewhat meaniningless here and is not divided by the resolution squared to create a higher threshold, which decreases the runtime
    gs.run_command(
        'r.watershed', 
        elevation="burneddem", threshold=config.thresholds.stream['local'] // (int(config.res[:-1]))**2, 
        accumulation="burnedaccum", drainage = "burnedraindir", 
        basin="burnedbasin", 
        flags='a', overwrite=True, memory = config.mem
    )
    gs.run_command("r.stream.extract", elevation= "burneddem",   threshold=config.thresholds.stream['local'] // (int(config.res[:-1]))**2,
                       accumulation="burnedaccum", stream_raster=  "burnedstreams")
    
    #create stream raster from flow grids
    gs.mapcalc('{0}_{2} = if({1} > 0, 1,0)'.format('r_flow_reaches','burnedstreams','local'))
    
    
    logger.info('created accumulation, drainage direction, and basins from burnedDEM')
    
    
    riverine_nodes_exists = True if config.riverine_flow_nodes.in_file.as_posix() != 'None' else False
    riverine_reaches_exists = True if config.riverine_flow_reaches.in_file.as_posix() != 'None' else False
    
    if riverine_nodes_exists:
        if riverine_reaches_exists:
            logger.info("importing fluvial streams")

            gs.run_command(
            'v.import', 
            input=config.root/config.riverine_flow_reaches.out_file.geojson, 
            output="tmp_flow_reaches", 
            extent="region", overwrite = True
            )

            gs.run_command(
            'v.clip', 
            input="tmp_flow_reaches",
            clip='huc12buff_95pct',
            output="flow_reaches_v", 
            overwrite = True
            )
            gs.run_command(
                'g.remove', 
                type='vector',
                name = 'tmp_flow_reaches',
                flags='f'
            )

            gs.run_command(
                'v.to.rast', 
                input='flow_reaches_v',
                output = 'r_flow_reaches_nonlocal',
                use='val',
                val=1
            )
            # burn known streams into the DEM
            logger.info("burning fluvial streams into DEM")
            gs.run_command(
                'r.carve', 
                output="burneddem_riverine",
                raster="dem", vector="flow_reaches_v", 
                depth="10", overwrite=True, flags="n"
            ) 
            logger.info('burned riverine streams into DEM')
        else:
            logger.info("burning fluvial streams into DEM")
            gs.run_command(
                'r.carve', 
                output="burneddem_riverine",
                raster="dem", vector="flowline", 
                depth="10", overwrite=True, flags="n"
            ) 
            logger.info('burned riverine streams into DEM')

        # calculate watershed properties (accumulation, drainage direction, basins)
        # for the burned DEM 
        gs.run_command(
            'r.watershed', 
            elevation="burneddem_riverine", threshold=config.thresholds.stream['nonlocal'] // (int(config.res[:-1]))**2, 
            accumulation="burnedaccum_riverine", drainage = "burnedraindir_riverine", 
            basin="burnedbasin_riverine", 
            flags='a', overwrite=True, memory = config.mem
        )
        
        #create non_local flow reaches          
        gs.run_command("r.stream.extract", elevation= "burneddem_riverine",   threshold=config.thresholds.stream['nonlocal'] // (int(config.res[:-1]))**2,
                   accumulation="burnedaccum_riverine", stream_raster=  "burnedstreams_riverine")
        #create stream raster from flow grids
        gs.mapcalc('{0}_{2} = if({1} > 0, 1,0)'.format('r_flow_reaches','burnedstreams_riverine','nonlocal'))
        logger.info('created riverine reaches from burnedDEM')
        
        #set variables based on riverine
        orig_dem = 'dem'
        dem = "burneddem_riverine"
        accum = 'burnedaccum_riverine'
        draindir = 'burnedraindir_riverine'
        basin = 'burnedbasin_riverine'
        r_streams = 'r_flow_reaches'
    
        
    else:
        #create non_local flow reaches          
        gs.run_command("r.stream.extract", elevation= "burneddem",   threshold=config.thresholds.stream['nonlocal'] // (int(config.res[:-1]))**2,
                   accumulation="burnedaccum", stream_raster=  "burnedstreams_nl")
        #create stream raster from flow grids
        gs.mapcalc('{0}_{2} = if({1} > 0, 1,0)'.format('r_flow_reaches','burnedstreams_nl','nonlocal'))
        logger.info('created riverine reaches from burnedDEM')
        
        #set variables based on pluvial
        orig_dem = 'dem'
        dem = "burneddem"
        accum = 'burnedaccum'
        draindir = 'burnedraindir'
        basin = 'burnedbasin'
        r_streams = 'r_flow_reaches'
                    

    if riverine_nodes_exists:
        logger.info("importing riverine flow nodes")

        gs.run_command(
        'v.import', 
        input=config.root/config.riverine_flow_nodes.out_file.geojson, 
        output="tmp_flow_points", 
        extent="region", overwrite = True
        )

        gs.run_command(
        'v.clip', 
        input="tmp_flow_points",
        clip='huc12buff_95pct',
        output="flow_points_v", 
        overwrite = True
        )
        gs.run_command(
            'g.remove', 
            type='vector',
            name = 'tmp_flow_points',
            flags='f'
        )
        
        riverine_heatmap_exists = None if config.FEMA_map_type == 'None' else config.FEMA_map_type
        if riverine_heatmap_exists:
            #create FEMA AEP grids
            logger.info("creating riverine floodplain grids")
            AEP_grid = create_FEMA_AEP_grids("huc12buff", config)
            gs.run_command('r.null', map=AEP_grid, null=0) 
            plot_grass_layer(AEP_grid, str(log_out_dir/"{}.png".format(AEP_grid)), vector=False)
        
    else: 
        logger.info("no riverine nodes found; proceeding without riverine functions")
        

    heatmap_exists = (config.root/config.heatmap.out_file.vrt).exists()
    if heatmap_exists:
        logger.info("importing heatmap")
        gs.run_command(
            'r.import', 
            input=config.root/config.heatmap.out_file.vrt, 
            output=("heatmap" + "_" + config.model_type), 
            extent = "region", overwrite=True, memory = config.mem
        ) 
    else: 
        logger.info("no heatmap file found; proceeding without creating heatmap")
       
    logger.info("starting layer calculations")
    
    #all types
    # calculate the topographic wetness index
    gs.run_command(
        'r.slope.aspect', 
        elevation=orig_dem, slope = "slope_tmp", 
        overwrite=True
    )
    #change slope of 0 to 0.01 degrees 
    gs.mapcalc('slope = if(slope_tmp == 0,0.01,slope_tmp)')
    
    gs.mapcalc('$twi = log($accum*pow($cell_size,2)/tan($slope), 2.718282)',
        twi="twi", 
        accum="burnedaccum",
        cell_size=cell_size, 
        slope="slope"
    )
    
    logger.info('created topographic wetness index (%s)'%("twi"))
    
    logger.info("plotting flow accumulation")
    accum_array = utils.mask_array(garray.array('MASK'), garray.array(accum))
    plot_accumulation(accum_array, log_out_dir/"accumulation.png")
    
    #create riverine flow reaches if applicable
    if riverine_nodes_exists:
        #create flow grid from flow nodes
        logger.info("creating riverine flow grids")
        flow_keys = list(config.name_events.keys()) 
        #snap flow nodes to stream raster within 100m
        gs.run_command(
            'r.stream.snap',
            input='flow_points_v',
            output='flow_points_v_snapped',
            stream_rast= r_streams+'_nonlocal',
            radius=100//int(config.res[:-1])
        )
        #reconnect to original database table
        gs.run_command('v.db.connect',map='flow_points_v_snapped',table='flow_points_v')
        
        flow_grids = create_flow_grids(flow_points = "flow_points_v_snapped",flow_keys =flow_keys,drain_dir = draindir,config = config)
        # fill in missing with 0 for later analysis
        for flow_grid in flow_grids:
            gs.run_command('r.null', map=flow_grid, null=0) 
        #create stream raster from flow grids
        gs.mapcalc('{1}_{2} = if({0} > 0, 1,0)'.format(flow_grids[0],r_streams,'nonlocal'))
        
#     else:
        
#         #pluvial
#         logger.info("starting calculations for jobs %s in parallel"%(jobs))
#         workers = multi.cpu_count()
#         logger.info("identified %s workers"%(workers))

#         mapcalc = Module("r.mapcalc", overwrite=True, run_=False)
#         rthin =  Module("r.stream.extract", overwrite=True, run_=False) 
#         # thin stream network -updated to use stream extract function
#         run_parallel(
#             rthin, workers, jobs, 
#             elevation=lambda job: dem, 
#             threshold=lambda job: config.thresholds.stream[job] // (int(config.res[:-1]))**2,
#             accumulation=accum,
#             stream_raster= lambda job: r_streams + "_tmp_" + job
#         )

#         #update to binary stream representation
#         for job in jobs:
#             gs.mapcalc('{0}_{1} = if({0}_tmp_{1} > 0, 1,0)'.format(r_streams,job))
#             plot_grass_layer(r_streams + "_" + job, str(log_out_dir/"streams_extract_thinned_{}.png".format(job)), vector=True)
#         plot_grass_layer(r_streams+',str(log_out_dir/"river_reaches.png"), vector=True)
#         logger.info("thinned stream network")

    logger.info("starting calculations for jobs %s in parallel"%(jobs))
    workers = multi.cpu_count()
    logger.info("identified %s workers"%(workers))
    rthin =  Module("r.stream.extract", overwrite=True, run_=False) 
    mapcalc = Module("r.mapcalc", overwrite=True, run_=False)
    rstreamdistance = Module("r.stream.distance", overwrite=True, run_=False, flags = "m", memory = config.mem)
    # calculate height and distance from drainage 
    run_parallel(
        rstreamdistance, workers, jobs, 
        direction = draindir,
        stream_rast= lambda job: r_streams + "_" + job,
        elevation= orig_dem,
        distance = lambda job: "distancestream" + "_" + job,
        difference = lambda job: "elevationstream" + "_" + job
    )
    logger.info("calculated height and distance from drainage")

    # calculate the contributing area 
    run_parallel(
        rstreamdistance, workers, jobs, 
        direction=draindir,
        stream_rast= lambda job: r_streams + "_" + job,
        elevation= accum,
        difference = lambda job: "contribarea" + "_" + job
    )
    logger.info("calculated raw contributing area")
#     for job in jobs:
#         plot_grass_layer(f"contribarea_{job}", str(log_out_dir/f"contributing_area_{job}.png"))

    # actual contrib area
    actual_contrib_area_exp = lambda job: "%s = if((-%s+%s)<%s,%s,-%s+%s)" %(
        "finalcontribarea"+'_'+job,
        "contribarea"+'_'+job,
        accum,
        config.thresholds.stream[job] // (int(config.res[:-1]))**2,
        config.thresholds.stream[job] // (int(config.res[:-1]))**2,
        "contribarea"+'_'+job,
        accum 
    )
    run_parallel(mapcalc, workers, jobs, expression = actual_contrib_area_exp)
    logger.info("calculated final contributing area")
    for job in jobs:
        plot_grass_layer(
            f"finalcontribarea_{job}", 
            str(log_out_dir/f"final_contributing_area_{job}.png")
        )

    # final stream elevation 
    final_stream_elevation_exp = lambda job: "%s = %s*if(%s<0, 0,1)+.01"% (
        "finalelevationstream"+'_'+job,  
        "elevationstream"+'_'+job,
        "elevationstream"+'_'+job
    )
    run_parallel(mapcalc, workers, jobs, expression = final_stream_elevation_exp)
    logger.info("calculated final stream elevation")
    for job in jobs:
        plot_grass_layer(
            f"elevationstream_{job}", 
            str(log_out_dir/f"final_stream_elevation_{job}.png"))
    
    # hydraulic radius: hr = cross-sectional area/wetted perimeter
    # assume a rectangular channel
    hr = lambda job: f"{'hr_' + job} =" \
        f"({'finalelevationstream_' + job} * {'distancestream_' + job}) " \
        f"/ (2 * {'finalelevationstream_' + job} + 2 * {'distancestream_' + job})" 

    run_parallel(mapcalc, workers, jobs, expression = hr)
    logger.info("calculated hydraulic radius")
    for job in jobs:
        plot_grass_layer(f"hr_{job}", str(log_out_dir/f"hr_{job}.png"))

    # calc average slope by multiplying thinned streams by non-zero slope
    stream_slope_exp = lambda job: f"streamslope_{job} = "+r_streams+f"_{job} * tan(slope)"
    run_parallel(mapcalc, workers, jobs, expression = stream_slope_exp)

#     #Calculating average slope by filtering over slopes at stream centerline. This would replace the global average slope used in the index calcs.
#     rneighbors = Module("r.neighbors", overwrite=True, run_=False)

#     # Use sum and then divide by the count to get the average. Do not use the average directly as this grass function rounds to an integer
#     run_parallel(
#         rneighbors, workers, jobs, 
#         input = lambda job: f"streamslope_{job}",
#         output= lambda job: f"streamslope_filtered_sum_{job}",
#         method = 'sum',
#         size = '5'
#     )

#     run_parallel(
#         rneighbors, workers, jobs, 
#         input = lambda job: f"streamslope_{job}",
#         output= lambda job: f"streamslope_filtered_count_{job}",
#         method = 'count',
#         size = '5'
#     )
    
#     stream_slope_avg_exp = lambda job: f"streamslope_filtered_{job} = streamslope_filtered_sum_{job} / streamslope_filtered_count_{job}"
#     run_parallel(mapcalc, workers, jobs, expression = stream_slope_avg_exp)

#     for job in jobs:
#         # fill in missing with 0, done for the difference calc with rstreamdistance in the next step
#         gs.run_command('r.null', map=f"streamslope_filtered_{job}", null=0) 

#     run_parallel(
#         rstreamdistance, workers, jobs, 
#         direction=draindir,
#         stream_rast= lambda job: r_streams + "_" + job,
#         elevation = lambda job: f"streamslope_filtered_{job}",
#         difference = lambda job: f"streamslope_filtered_stream_{job}"
#     )
    
#     final_slope_exp = lambda job: f"final_slope_{job} = if(streamslope_filtered_stream_{job} < 0, -1*streamslope_filtered_stream_{job}, streamslope_filtered_{job})"
    
#     run_parallel(mapcalc, workers, jobs, expression = final_slope_exp)

#     for job in jobs:
#         plot_grass_layer(f"final_slope_{job}", str(log_out_dir/f"final_slope_{job}.png"))

    #Superceeded by average slope along the stream and mapping back to cross section points.
    avg_slopes = {}
    for job in jobs:
        mask = garray.array("MASK")
        streamslope_arr = utils.mask_array(mask, garray.array(f"streamslope_{job}"))
        avg_slopes[job] = np.nanmean(streamslope_arr)
        logger.info(f"avg slope for {job}: {avg_slopes[job]}")

    #calculate mannings velocity: 1.49/n * hr^(2/3) * s^(1/2) [SI] or 1.0/n * hr^(2/3) * s^(1/2) [metric]
    #want mannings velocity from center of closest stream using the average slope 
    #for local & non-local streams

    # get manning's n from closest stream
    mannings_n_stream = lambda job: f"mannings_n_{job} = "+r_streams+f"_{job} * mannings_n"
    run_parallel(mapcalc, workers, jobs, expression = mannings_n_stream)

    for job in jobs:
        # fill in missing with 0
        gs.run_command('r.null', map=f"mannings_n_{job}", null=0) 
        plot_grass_layer(f"mannings_n_{job}", str(log_out_dir/f"mannings_n_{job}.png"))

    run_parallel(
        rstreamdistance, workers, jobs, 
        direction=draindir,
        stream_rast= lambda job: r_streams + "_" + job,
        elevation = lambda job: f"mannings_n_{job}",
        difference = lambda job: f"final_mannings_n_{job}"
    )

    final_mannings_n_exp = lambda job: f"final_mannings_n_{job} = if(final_mannings_n_{job} < 0, -1*final_mannings_n_{job}, mannings_n_{job})"
    run_parallel(mapcalc, workers, jobs, expression = final_mannings_n_exp)

    for job in jobs:
        plot_grass_layer(f"final_mannings_n_{job}", str(log_out_dir/f"final_mannings_n_{job}.png"))

    mannings_vel= lambda job: f"{'manningsvel_' + job} =" \
        f"1.0/final_mannings_n_{job} * exp({'hr_' + job}, 2/3) * exp({avg_slopes[job]}, 0.5)"

    run_parallel(mapcalc, workers, jobs, expression = mannings_vel)

    for job in jobs:
        plot_grass_layer(f"manningsvel_{job}", str(log_out_dir/f"mannings_vel_{job}.png"))
    

    for job in jobs:
        
        #riverine
        if job == 'nonlocal':
            if riverine_nodes_exists:
                for flow_key in flow_keys:
                    flow_interval = config.name_events[flow_key]
                    flow_grid = "f_drain_flow_{0}".format(flow_interval)
                    gs.run_command("r.stream.distance", 
                                   direction=draindir,
                                   stream_rast= r_streams+'_'+job,
                                   elevation = flow_grid,
                                   difference = "final_{0}".format(flow_grid),
                                   memory = config.mem,
                                   flags='m'
                                  )

                    final_flow = gs.run_command("r.mapcalc",expression="final_{0} = \
                                                                        if(final_{0} < 0, \
                                                                        -1*final_{0}, f_drain_flow_{1} \
                                                                           )".format(flow_grid, flow_interval))
                    plot_grass_layer("final_{0}".format(flow_grid), str(log_out_dir/"{0}.png".format(flow_grid)), vector=False)

                    # riverine flood index
                    riverine_flood_index_exp = gs.run_command("r.mapcalc",expression="riverinefloodindex_{0}_{1} = log(final_{0} / ((if(distancestream_{1} * finalelevationstream_{1} == 0,{2}*0.01,distancestream_{1}) * finalelevationstream_{1}) * manningsvel_{1}), 2.718282)".format(flow_grid,job,int(config.res[:-1])))
                    plot_grass_layer("riverinefloodindex_{0}_{1}".format(flow_grid,job), str(log_out_dir/"riverinefloodindex_{0}_{1}.png".format(flow_grid,job)), vector=False)  
                    
                    # #water level
                    # water_level = gs.run_command("r.mapcalc",expression="water_level_{0}_{1} = exp(final_mannings_n_{1},3) * exp(final_{0},3) * exp(,0.5) / ((if(distancestream_{1} * finalelevationstream_{1} == 0,{2}*0.01,distancestream_{1}) * finalelevationstream_{1}) * manningsvel_{1}), 2.718282)".format(flow_grid,job,int(config.res[:-1])))
                    # plot_grass_layer("riverinefloodindex_{0}_{1}".format(flow_grid,job), str(log_out_dir/"riverinefloodindex_{0}_{1}.png".format(flow_grid,job)), vector=False)  
            else:
                # flood index
                geo_flood_index_exp = lambda job: f"geofloodindex_{job} = " \
                    f"log(finalcontribarea_{job}" \
                    f"/ (distancestream_{job} * finalelevationstream_{job} * manningsvel_{job}), " \
                    "2.718282)"

                run_parallel(mapcalc, workers, jobs, expression = geo_flood_index_exp)

            logger.info("calculated non_local flood index")
        else:
            # flood index
            geo_flood_index_exp = lambda job: f"geofloodindex_{job} = " \
                f"log(finalcontribarea_{job}" \
                f"/ (distancestream_{job} * finalelevationstream_{job} * manningsvel_{job}), " \
                "2.718282)"

            run_parallel(mapcalc, workers, jobs, expression = geo_flood_index_exp)
            logger.info("calculated local flood index")
    

    logger.info("all calculations complete. GRASS has the following rasters and vectors: " + 
                 str(gs.list_strings(type=['raster', 'vector'], mapset=mapset)))
    

    logger.info("prepping data for ML")

    # use huc12 as mask for plots (not buffered layer)
    gs.run_command('r.mask', vector="huc12", overwrite = True)
    mask = garray.array("MASK")
    
    # create list of layers to add to final dataframe 
    layers_for_df = []
    
    if heatmap_exists:
        layers_for_df.append("heatmap" + "_" + config.model_type)
        

    if riverine_nodes_exists:
        if riverine_heatmap_exists:
            #floodplain raster map
            layers_for_df.append("heatmap_riverine")
            fig, axs = plt.subplots(1, 1, figsize=(15,15))
            im = axs.imshow(utils.mask_array(mask,  garray.array("heatmap_riverine")))
            axs.set_title("riverine AEP grids from FEMA")
            plt.colorbar(im, ax=axs)
            plt.suptitle('riverine floodplains')
            plt.savefig(os.path.join(log_out_dir,"output_rasters_%s.png"%('riverine floodplain')))
        
        layers_for_df.append("twi")
        for job in jobs:
            layers_for_df.append("finalelevationstream"+'_'+job)
            layers_for_df.append("elevationstream"+'_'+job)
            layers_for_df.append("distancestream"+'_'+job)
            layers_for_df.append("finalcontribarea"+'_'+job)
        
        #riverine flows
        fig, axs = plt.subplots((len(flow_keys)+1)//2, 2, figsize=(15,7.5*(len(flow_keys)+1)//2))
        for flow_key in flow_keys:
            flow_interval = config.name_events[flow_key]
            flow_grid = "f_drain_flow_{0}".format(flow_interval)
            layer = "final_{0}".format(flow_grid)
            #layers_for_df.append(layer)
            #index non local only
            indx_layer = "riverinefloodindex_{0}_{1}".format(flow_grid,'nonlocal')    
            layers_for_df.append(indx_layer)

            #plot
            row = flow_keys.index(flow_key)//2
            col = flow_keys.index(flow_key)%2
            if row > 0:
                im = axs[row,col].imshow(utils.mask_array(mask,  garray.array(layer)))
                axs[row,col].set_title("final riverine flows - {} yr event".format(flow_interval))
                plt.colorbar(im, ax=axs[row,col])
            else:
                im = axs[col].imshow(utils.mask_array(mask,  garray.array(layer)))
                axs[col].set_title("final riverine flows - {} yr event".format(flow_interval))
                plt.colorbar(im, ax=axs[col])
            plt.suptitle('riverine flows')
            plt.savefig(os.path.join(log_out_dir,"output_rasters_%s.png"%('riverine flows')))
            

            
    else:
        layers_for_df.append("twi")
        for job in jobs:
            layers_for_df.append("geofloodindex"+'_'+job)
            layers_for_df.append("finalelevationstream"+'_'+job)
            layers_for_df.append("elevationstream"+'_'+job)
            layers_for_df.append("distancestream"+'_'+job)
            layers_for_df.append("finalcontribarea"+'_'+job)

            logger.info("saving images of arrays for job "  + job)
            fig, axs = plt.subplots(2, 2, figsize=(15,15))
            im = axs[0, 0].imshow(utils.mask_array(mask,  garray.array("finalelevationstream"+'_'+job)), cmap = "terrain")
            axs[0, 0].set_title("final stream elevation")
            plt.colorbar(im, ax=axs[0, 0])
            im = axs[0, 1].imshow(utils.mask_array(mask,  garray.array("geofloodindex"+'_'+job)))
            axs[0, 1].set_title("geo flood index")
            plt.colorbar(im, ax=axs[0, 1])
            im = axs[1, 0].imshow(utils.mask_array(mask, garray.array("twi")))
            axs[1, 0].set_title("topographic wetness index")
            plt.colorbar(im, ax=axs[1, 0])
            im = axs[1, 1].imshow(utils.mask_array(mask, garray.array("distancestream"+'_'+job)))
            axs[1, 1].set_title("stream distance")
            plt.colorbar(im, ax=axs[1, 1])
            plt.suptitle(job)
            plt.savefig(os.path.join(log_out_dir, "output_rasters_%s.png"%(job)))
        
    
    # Set up Spark Session with delta lake
    spark = utils.get_spark_storage(str(config.grass_data.storage),os.environ["AZURE_CLIENT_ID"],os.environ["AZURE_CLIENT_SECRET"],os.environ["AZURE_TENANT_ID"])

    df = layers_to_spark_df(huc12, layers_for_df, mask, gs, spark, tile_size=256)
    
    if os.environ["APP_ENV"].lower() == "local":
        path = str(config.root/config.grass_data.out_file.delta)
        delta_table_exists = (config.root/config.grass_data.out_file.delta).exists()
        huc_partition_exists = (config.root/config.grass_data.out_file.delta/("huc=" + huc12)).exists()
    elif os.environ["APP_ENV"].lower() == "cloud":
        path = "abfss://{}@{}.dfs.core.windows.net/{}".format(
            str(config.grass_data.container), str(config.grass_data.storage), str(config.grass_data.out_file.delta))
        
        grass_blob_client = utils.connect_to_blob(config.grass_data.storage, config.grass_data.container, config.grass_data.out_file.delta, credentials)
        delta_table_exists = grass_blob_client.exists()
        container_client = utils.connect_to_container(config.grass_data.storage, config.grass_data.container, credentials)
        blob_list = container_client.list_blobs(name_starts_with=config.grass_data.out_file.delta)
        huc_partition_exists = False
        for blob in blob_list:
            if ("huc=" + huc12) in blob.name:
                huc_partition_exists = True
                break
            else:
                continue        
        
        logger.info("copying logs to blob storage")
        for log_file in os.listdir(log_out_dir):
            logger.info("writing log %s to blob"%(log_file))
            utils.upload_blob(
                os.path.join(log_out_dir, log_file), 
                config.log.storage, 
                config.log.container,
                os.path.join(("huc" + huc12), log_file), 
                credentials,
                overwrite=True
            )
    
    # update/append to delta table 
    logger.info("dataframe columns:")
    logger.info(df.columns)

    if huc_partition_exists:
        logger.info("overwriting huc partition")
        df.write.format("delta") \
            .partitionBy('huc', "idx_0", "idx_1") \
            .option("mergeSchema", "true") \
            .mode("overwrite") \
            .option("replaceWhere", "huc == '" + huc12 + "'") \
            .save(path)
    else: 
        if delta_table_exists: 
            logger.info("appending to delta")
        else: 
            logger.info("creating delta table")
            
        df.write.format("delta") \
            .option("mergeSchema", "true") \
            .mode("append") \
            .partitionBy('huc', "idx_0", "idx_1") \
            .save(path)

    #spark.stop()
    
            
if __name__ == '__main__':
    logging.root.setLevel(logging.ERROR)
    logging.basicConfig(format = '[%(asctime)s] [%(levelname)s] [%(module)s] : %(message)s')

    parser = argparse.ArgumentParser()
    loc = parser.add_mutually_exclusive_group(required=True)
    loc.add_argument('--point', '-pt', help="a point in lat/long coordinates", nargs=2, type=float)
    loc.add_argument('--huc12', '-huc', help="a huc12 code", type=str)

    args = parser.parse_args()
    
    if args.point:
        huc12 = utils.get_huc12(tuple(args.point))
    else:
        huc12 = args.huc12

    config = create_config(huc12)
    
    main(huc12=huc12, config=config)
    