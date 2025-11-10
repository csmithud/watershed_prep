from src.grass_functions import *

class GrassWatershed:
    
    #set project level variables
    Project_Area = 'nebraska_regression_stantec'
    sr = '26852' #set to None if you want to use the DEM's original projection
    res = '10m' #meters #DEM resolution, options are '1m', '3m', '10m', '30m', 'OPR'
    buffer = 2000 #buffer around the project area in native units
    #buffer = 0 #set to 0 if you do not want to buffer the project area
    Location = Project_Area+'_'+sr #initiate variable here
    Mapset = 'PERMANENT' #always set to this
    threshold = 1 #sq mi drainage area
    max_mem = psutil.virtual_memory().available / 1000000
    
    #dem info
    dem_preprocessed = False
    dem_base_name = 'state_dem' #for saving in grass
    aligned = False
    carved = True
    
    #standard folder structure parameters
    data_dir = pl.Path(os.getcwd()).parent/'data'
    vector_dir = data_dir/'Vectors'/Project_Area
    raster_dir = data_dir/'Rasters'/Project_Area
    

    #if you'd like to clean up after yourself, set this to True
    auto_delete = False
    
    def create_output_dirs(self):
        if not os.path.exists(GrassWatershed.vector_dir):
            os.makedirs(GrassWatershed.vector_dir)
        if not os.path.exists(GrassWatershed.raster_dir):
            os.makedirs(GrassWatershed.raster_dir)
    
    
    #watershed specific
    def __init__(self,data_scale,analysis_scale,aoi, geometry):
        ## Set variables analysis
        self.geometry = geometry #point of watershed outlet or polygon watershed boundary
        self.data_scale = data_scale #other options HUC12, HUC10, HUC8 OR the field name of the source data to be split <-- case sensitive
        self.analysis_scale = analysis_scale #Can be smaller than the data scale
        self.aoi = aoi #     # Value within the data_scale field used for data selection
        self.basins = GrassWatershed.vector_dir/f'aoi_{self.aoi}_basin.geojson' # provide geojson with polygons or the code will create it
        self.outlet = GrassWatershed.vector_dir/f'aoi_{self.aoi}_outlet.geojson' # provide geojson with polygons or the code will create it with a point
        self.to_headwaters = False #False if only interested in local drainage area
        
    
    #hone in on the area of importance for loops
    def set_grass_selection(self):
        '''
        
        '''
        if self.data_scale.find('HUC') >= 0:
            huc_level = re.findall("[0-9]+",self.data_scale)[0]
            if huc_level == str(len(aoi)):
                select_data = self.aoi
            else:
                select_data = self.aoi[:int(huc_level)]
        else:
            select_data = self.aoi
        self.select_data = select_data
        print(f'base data is {self.select_data}, analysis area is {self.aoi}')

    def set_dem_name(self):
        '''
        
        '''
        if GrassWatershed.dem_preprocessed:
            dem = f'{GrassWatershed.dem_base_name}_{self.select_data}'
            if aligned:
                dem +='_a'
            if carved:
                dem +='_b'
        else:
            dem = GrassWatershed.dem_base_name
        self.dem = dem
        assert self.dem+"@PERMANENT" in self.grass_maps['raster'], 'Check that dem exists in GRASS map layers'

    def assign_grass_variables(self):
        #initiate variables
        self.accum = f'accum_{self.select_data}'
        self.drain_dir = f'drain_dir_{self.select_data}'
        self.r_basins = f'r_basins_{self.select_data}'
        self.v_basins = f'v_basins_{self.select_data}'
        self.sub_basins = f'subbasins_{self.select_data}'
        self.v_streams = f'v_streams_{self.select_data}'
        self.r_streams = f'r_streams_{self.select_data}'
        self.v_outlet = f'v_outlet_{self.select_data}'
        self.r_outlet = f'r_outlet_{self.select_data}'
    
    def list_existing_grass(self,print_it=True):
        #List Existing Files: Vectors and Rasters
        layers = {'vector':[],'raster':[]}
        if print_it:
            print('Available vector maps:')
        for vect in gs.list_strings(type='vector'):
            if print_it:
                print (vect)
            layers['vector'].append(vect)

        if print_it:
            print('\nAvailable raster maps:')
        for rast in gs.list_strings(type='raster'):
            if print_it:
                print (rast)
            layers['raster'].append(rast)
        self.grass_maps = layers

    def get_grass_grid_size(self):
        '''
        '''
        reg = gs.parse_command('g.region', raster=self.dem, flags='pgm',align=self.dem)
        cell_area = (float(reg.nsres)*float(reg.ewres))
        if float(reg.nsres)/float(GrassWatershed.res[:-1]) > 2: 
            #assum data is in feet
            self.hunits = 'ft'
        else:
            self.hunits = 'm'
        self.cell_area = cell_area
    

    
    def get_threshold(self):
        '''
        '''
        if self.hunits == 'ft':
            self.threshold = GrassWatershed.threshold*27878400/self.cell_area
            self.search_radius = 500*3.28084/np.sqrt(self.cell_area) #assume search radius of 500m
        else:
            self.threshold = GrassWatershed.threshold*2590136.755/self.cell_area
            self.search_radius = 500/np.sqrt(self.cell_area) #assume search radius of 500m
        
    def reset_proj_region(self):
        '''
        '''
        self.list_existing_grass(print_it=False)
        #remove mask if it exists and reset region
        if 'MASK@PERMANENT' in self.grass_maps['raster']:
            gs.run_command('r.mask',flags = 'r')
        gs.run_command('g.region',raster = self.dem,align=self.dem,zoom=self.dem)
    
    def get_rough_watershed_data(self,resample=10,overwrite=False):
        '''
        '''
        if 'MASK@PERMANENT' in self.grass_maps['raster']:
            gs.run_command('r.mask',flags = 'r')
        gs.run_command('g.region',raster = self.dem,align=self.dem,zoom=self.dem)
        gs.run_command('g.region',res = resample*np.sqrt(self.cell_area))
        self.get_threshold()
        self.threshold_rough = self.threshold/resample
        self.search_radius_rough = self.search_radius/10
        if f'r_streams_{GrassWatershed.Project_Area}_rough'+"@PERMANENT" in self.grass_maps['raster'] and not overwrite:
            print('using existing project-wide watershed data for the project area')
            self.rough_created = True
        else:
            self.rough_created = False
            print('creating  project-wide watershed data for the project area')
        assert self.geometry in ('point','polygon'), 'Geometry type must be a point or polygon'
        if self.geometry == 'point' and not self.rough_created:
            #run at x times resolution to get initial boundary area
            
            #get streams to snap to
            gs.run_command('r.watershed', elevation=self.dem, accumulation = f'accum_{GrassWatershed.Project_Area}_rough',
                           flags= 'sabm',memory = GrassWatershed.max_mem) ##note that this is in feet
            
            gs.run_command('r.stream.extract', elevation=self.dem, accumulation = f'accum_{GrassWatershed.Project_Area}_rough', threshold =self.threshold_rough, 
                           direction= f'drain_dir_{GrassWatershed.Project_Area}_rough',
                           stream_raster = f'r_streams_{GrassWatershed.Project_Area}_rough',stream_vector = f'v_streams_{GrassWatershed.Project_Area}_rough',
                           memory = GrassWatershed.max_mem)
        
           
    def get_basin_area(self,overwrite=False):
        '''
        '''
        assert self.geometry in ('point','polygon'), 'Geometry type must be a point or polygon'
        if self.geometry == 'point':
            gs.run_command('g.region',raster=f'r_streams_{GrassWatershed.Project_Area}_rough')
            gs.run_command('v.import', input= self.outlet,  output= f'aoi_{self.select_data}_outlet_raw')
            
            gs.run_command('r.stream.snap',input=f'aoi_{self.select_data}_outlet_raw', output=f'{self.v_outlet}_rough', 
                           stream_rast=f'r_streams_{GrassWatershed.Project_Area}_rough',accumulation=f'accum_{GrassWatershed.Project_Area}_rough', 
                           threshold = self.threshold_rough, radius = self.search_radius_rough,memory = GrassWatershed.max_mem)
            #save as rough polygon boundary
            gs.run_command('r.stream.basins',direction = f'drain_dir_{GrassWatershed.Project_Area}_rough', points = f'{self.v_outlet}_rough',
                           basins =f'{self.r_basins}_rough', memory = GrassWatershed.max_mem)
            gs.run_command('r.to.vect', input=f'{self.r_basins}_rough', output=f'aoi_{self.select_data}_aoi_raw', type='area',flags='s')
            gs.run_command('v.out.ogr', input= f'aoi_{self.select_data}_aoi_raw' ,type = 'area',output = str(self.basins).replace('.geojson','_rough.geojson'), 
                           format = 'GeoJSON')
            gs.run_command('v.out.ogr', input=  f'aoi_{self.select_data}_outlet_raw' ,type = 'point',output = 'aoi.geojson', format = 'GeoJSON')
        else:
            #refined delineation
            #import area of interest
            temp = gpd.read_file(self.basins)
            aoi_path = str(self.basins).replace('.geojson','_rough.geojson')
            temp.loc[temp[self.analysis_scale] == self.aoi].to_file(aoi_path, driver="GeoJSON")
            gs.run_command('v.import', input= aoi_path,  output= f'aoi_{self.select_data}_aoi_raw')

        self.reset_proj_region()

        print('added basin aoi to grass')
        #buffer to create potentital watershed area
        #add mask and reset region
        gs.run_command('v.buffer',input=f'aoi_{self.select_data}_aoi_raw',output=f'aoi_{self.select_data}_aoi_raw_buffer',distance = GrassWatershed.buffer)
        #mask outside cells
        gs.run_command('r.mask',vector = f'aoi_{self.select_data}_aoi_raw_buffer',overwrite=True)
        #set the region based on the buffer but align with DEM raster
        gs.run_command('g.region',raster = self.dem,align=self.dem,zoom=self.dem)

        #get outlet point by identifying the highest accumulation value along the perimeter.
        gs.run_command('r.watershed', elevation=self.dem, drainage = self.drain_dir, accumulation = self.accum, flags= 'sabm',memory = GrassWatershed.max_mem) ##note that this is in feet
        gs.run_command('v.rast.stats', raster=self.accum, map=f'aoi_{self.select_data}_aoi_raw',method='max',column='accum')
        max_accum = list(gs.parse_command('v.db.select', columns='accum_maximum',map = f'aoi_{self.select_data}_aoi_raw',flags='c').keys())
        for outlet_accum in max_accum:
            gs.run_command('r.mapcalc',expression = f'{self.r_outlet} = if({self.accum} == {outlet_accum},{self.select_data},null())')
            gs.run_command('r.to.vect', input=self.r_outlet, output=self.v_outlet, type='point')
            gs.run_command('v.out.ogr', input=  f'aoi_{self.select_data}_aoi_raw' ,type = 'point',output = self.outlet, format = 'GeoJSON')
            
        print("Delineating the watersheds")
        gs.run_command('r.stream.basins',direction = self.drain_dir, points = self.v_outlet,basins =self.r_basins, memory = GrassWatershed.max_mem)
        gs.run_command('r.stream.extract', elevation=self.dem, accumulation = self.accum, threshold =self.threshold, direction= self.drain_dir,
               stream_raster = self.r_streams,stream_vector = self.v_streams, memory = GrassWatershed.max_mem)
        print("Converting the delineated watershed rasters to vectors")
        gs.run_command('r.to.vect', input=self.r_basins, output= self.v_basins, type="area", flags='s')
        gs.run_command('v.out.ogr', input=  self.v_basins ,type = 'area',output = self.basins, format = 'GeoJSON')
        

