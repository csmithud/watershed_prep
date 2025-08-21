import yaml
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Optional
import dacite 
import pathlib 
import re
from collections import OrderedDict 
import platform

def create_config(huc:str):
    
    """
    Cerates a Configuration object to be used throughout the code to 
    get config parameters.
    
    :param dict config_yaml: path to YAML file to use for configuration, e.g. config.yaml
    :param huc: the HUC of interest
    """
    huc_level = len(huc)
    
    #if platform.system()
    try:
        os.environ["APP_ENV"]
    except KeyError:
        print("ENV environment variable not defined! please set to one of 'LOCAL' or 'CLOUD'")
        raise 
        
    assert os.environ["APP_ENV"] in ["LOCAL", "CLOUD"], "ENV environment variable not correct option; please set to one of 'LOCAL' or 'CLOUD'"

    if os.environ["APP_ENV"].lower() == "cloud":
        try: 
            os.environ["GRASS_BLOB_KEY"]
            os.environ["AZURE_CLIENT_ID"] 
            os.environ["AZURE_CLIENT_SECRET"] 
            os.environ["AZURE_TENANT_ID"] 
        except KeyError as err:
            print("missing required environment variable for Azure connections.", 
                  "Add necessary environment variables, or update APP_ENV to LOCAL for local development.")
            raise
    
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)

    with open((parentdir + "/config.{}.yaml").format(os.environ["APP_ENV"].lower())) as stream:
        try:
            raw_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        
    if huc_level not in [12,10,8]:
        raise ValueError("HUC number must be 8, 10, or 12 characters.") 
    
    def clean_file(file, huc):
        file = str(file)
        file = re.sub("(?i)%shuc2", huc[:2], file)
        file = re.sub("(?i)%shuc4", huc[:4], file)
        file = re.sub("(?i)%shuc8", huc[:8], file)
        file = re.sub("(?i)%shuc", huc, file)
        file = re.sub("(?i)%ssr", raw_config["sr"], file)
        file = re.sub("(?i)%sres", raw_config["res"], file)
                                                        
        assert "%" not in file, \
            "unable to fill in filename completely! after cleaning file is % after cleaning "%(file)
        return pathlib.Path(file)

    def validate_sr(sr):
        if type(sr) != str:
            raise ValueError("spatial reference (sr) must be string")
      
    def validate_res(res, dem_products):
        if res not in dem_products.keys():
            raise ValueError("resolution not in dem products available")
            
    def validate_dem(dem):
        dem_files = dem.out_file.valid_files().values()
        if not all([raw_config["res"] in str(file) for file in dem_files]):
            raise ValueError("resolution must be specified all DEM file names")
                   
    @dataclass
    class Thresholds:
        stream: dict
            
    @dataclass
    class OutFile:
        geojson: Optional[pathlib.Path] 
        parquet: Optional[pathlib.Path] 
        vrt: Optional[pathlib.Path]
        gdb: Optional[pathlib.Path] 
        delta: Optional[pathlib.Path] 
        tiff: Optional[pathlib.Path] 
            
        def __post_init__(self):
            self.geojson = clean_file(self.geojson, huc)
            self.parquet = clean_file(self.parquet, huc)
            self.vrt = clean_file(self.vrt, huc)
            self.gdb = clean_file(self.gdb, huc)
            self.delta = clean_file(self.delta, huc)
            self.tiff = clean_file(self.tiff, huc)
            
        def valid_files(self):
            return {k:v for k,v in self.__dict__.items() if v != pathlib.Path("None")}
            
    @dataclass
    class FileConfiguration:
        out_file: OutFile
        in_file: Optional[pathlib.Path]
        
        if os.environ["APP_ENV"] == "CLOUD":
            storage: Optional[str]
            container: Optional[str]

        def __post_init__(self):
            self.in_file = clean_file(self.in_file, huc)

    @dataclass
    class LogConfiguration:
        if os.environ["APP_ENV"] == "CLOUD":
            storage: str
            container: str
        else: 
            dir: pathlib.Path
                
    @dataclass 
    class Configuration:
        log: LogConfiguration
        sr: str
        res: str
        vert_unit: str
        mem: int
        huc_buff: int 
        thresholds: dict
        root: pathlib.Path
        gisdb: str
        pour_points: FileConfiguration
        huc_bounds: FileConfiguration
        nlcd:FileConfiguration
        nhd: FileConfiguration
        dem: FileConfiguration
        dem_products: dict
            
        def __post_init__(self):
            validate_sr(self.sr)
            validate_dem(self.dem)
            self.dem_products = OrderedDict(self.dem_products)
            
            
    converters = {
        None: None,
        pathlib.Path: pathlib.Path
    }
                                        
    __config = dacite.from_dict(
        data_class=Configuration, 
        data=raw_config, 
        config=dacite.Config(type_hooks=converters)
    )
    
    return __config
