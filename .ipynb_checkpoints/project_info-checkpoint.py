from src.grass_functions import *

class ProjectInformation:
    
    def __init__(self,Project_Area,sr,res,dem_preprocessed,dem_base_name,aligned,carved):
        #set project level variables
        self.Project_Area = Project_Area
        self.sr = sr #set to None if you want to use the DEM's original projection
        self.res = res #meters #DEM resolution, options are '1m', '3m', '10m', '30m', 'OPR'
        self.buffer = 2000 #buffer around the project area in native units
        self.Location = Project_Area+'_'+sr #initiate variable here
        self.Mapset = 'PERMANENT' #always set to this
        self.threshold = 1 #sq mi drainage area
        self.max_mem = psutil.virtual_memory().available / 1000000

        #dem info
        self.dem_preprocessed = dem_preprocessed
        self.dem_base_name = dem_base_name #for saving in grass
        self.aligned = aligned
        self.carved = carved

        #standard folder structure parameters
        self.data_dir = pl.Path(os.getcwd()).parent/'data'
        self.vector_dir = self.data_dir/'Vectors'/Project_Area
        self.raster_dir = self.data_dir/'Rasters'/Project_Area
    
    def create_output_dirs(self):
        if not os.path.exists(self.vector_dir):
            os.makedirs(self.vector_dir)
        if not os.path.exists(self.raster_dir):
            os.makedirs(self.raster_dir)


class GrassWatershed:

    #watershed specific
    def __init__(self,project, data_scale,analysis_scale,aoi, geometry):
        ## Set variables analysis
        self.project = project
        self.geometry = geometry #point of watershed outlet or polygon watershed boundary
        self.data_scale = data_scale #other options HUC12, HUC10, HUC8 OR the field name of the source data to be split <-- case sensitive
        self.analysis_scale = analysis_scale #Can be smaller than the data scale
        self.aoi = aoi #     # Value within the data_scale field used for data selection
        self.basins = self.project.vector_dir/f'aoi_{self.aoi}_basin.geojson' # provide geojson with polygons or the code will create it
        self.outlet = self.project.vector_dir/f'aoi_{self.aoi}_outlet.geojson' # provide geojson with polygons or the code will create it with a point
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
        if self.project.dem_preprocessed:
            dem = f'{self.project.dem_base_name}_{self.select_data}'
            if aligned:
                dem +='_a'
            if carved:
                dem +='_b'
        else:
            dem = self.project.dem_base_name
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
        self.r_streams_order = f'r_streams_order{self.select_data}'
    
    def reset_proj_region(self):
        '''
        '''
        self.list_existing_grass(print_it=False)
        #remove mask if it exists and reset region
        if 'MASK@PERMANENT' in self.grass_maps['raster']:
            gs.run_command('r.mask',flags = 'r')
        gs.run_command('g.region',raster = self.dem,align=self.dem,zoom=self.dem)
    
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
        if float(reg.nsres)/float(self.project.res[:-1]) > 2: 
            #assum data is in feet
            self.hunits = 'ft'
        else:
            self.hunits = 'm'
        self.cell_area = cell_area
    

    
    def get_threshold(self):
        '''
        '''
        if self.hunits == 'ft':
            self.threshold = self.project.threshold*27878400/self.cell_area
            self.search_radius = 500*3.28084/np.sqrt(self.cell_area) #assume search radius of 500m
        else:
            self.threshold = self.project.threshold*2590136.755/self.cell_area
            self.search_radius = 500/np.sqrt(self.cell_area) #assume search radius of 500m
        
    
    def get_rough_watershed_data(self,resample=10,overwrite=False):
        '''
        '''
        self.list_existing_grass(print_it=False)
        if 'MASK@PERMANENT' in self.grass_maps['raster']:
            gs.run_command('r.mask',flags = 'r')
        gs.run_command('g.region',raster = self.dem,align=self.dem,zoom=self.dem)
        gs.run_command('g.region',res = resample*np.sqrt(self.cell_area))
        self.get_threshold()
        self.threshold_rough = self.threshold/resample
        self.search_radius_rough = self.search_radius/10
        if f'r_streams_{self.project.Project_Area}_rough'+"@PERMANENT" in self.grass_maps['raster'] and not overwrite:
            print('using existing project-wide watershed data for the project area')
            self.rough_created = True
        else:
            self.rough_created = False
            print('creating  project-wide watershed data for the project area')
        assert self.geometry in ('point','polygon'), 'Geometry type must be a point or polygon'
        if self.geometry == 'point' and not self.rough_created:
            #run at x times resolution to get initial boundary area
            
            #get streams to snap to
            gs.run_command('r.watershed', elevation=self.dem, accumulation = f'accum_{self.project.Project_Area}_rough',
                           flags= 'sabm',memory = self.project.max_mem) ##note that this is in feet
            
            gs.run_command('r.stream.extract', elevation=self.dem, accumulation = f'accum_{self.project.Project_Area}_rough', threshold =self.threshold_rough, 
                           direction= f'drain_dir_{self.project.Project_Area}_rough',
                           stream_raster = f'r_streams_{self.project.Project_Area}_rough',stream_vector = f'v_streams_{self.project.Project_Area}_rough',
                           memory = self.project.max_mem)
        
           
    def get_basin_area(self,overwrite=False):
        '''
        '''
        assert self.geometry in ('point','polygon'), 'Geometry type must be a point or polygon'
        if self.geometry == 'point':
            gs.run_command('g.region',raster=f'r_streams_{self.project.Project_Area}_rough')
            gs.run_command('v.import', input= self.outlet,  output= f'aoi_{self.select_data}_outlet_raw')
            
            gs.run_command('r.stream.snap',input=f'aoi_{self.select_data}_outlet_raw', output=f'{self.v_outlet}_rough', 
                           stream_rast=f'r_streams_{self.project.Project_Area}_rough',accumulation=f'accum_{self.project.Project_Area}_rough', 
                           threshold = self.threshold_rough, radius = self.search_radius_rough,memory = self.project.max_mem)
            #save as rough polygon boundary
            gs.run_command('r.stream.basins',direction = f'drain_dir_{self.project.Project_Area}_rough', points = f'{self.v_outlet}_rough',
                           basins =f'{self.r_basins}_rough', memory = self.project.max_mem)
            gs.run_command('r.to.vect', input=f'{self.r_basins}_rough', output=f'aoi_{self.select_data}_aoi_raw', type='area',flags='s')
            gs.run_command('v.out.ogr', input= f'aoi_{self.select_data}_aoi_raw' ,type = 'area',output = str(self.basins).replace('.geojson','_rough.geojson'), 
                           format = 'GeoJSON')
            gs.run_command('v.out.ogr', input=  f'aoi_{self.select_data}_outlet_raw' ,type = 'point',output = str(self.outlet).replace('.geojson','_rough.geojson'), format = 'GeoJSON')
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
        gs.run_command('v.buffer',input=f'aoi_{self.select_data}_aoi_raw',output=f'aoi_{self.select_data}_aoi_raw_buffer',distance = self.project.buffer)
        #mask outside cells
        gs.run_command('r.mask',vector = f'aoi_{self.select_data}_aoi_raw_buffer',overwrite=True)
        #set the region based on the buffer but align with DEM raster
        gs.run_command('g.region',raster = self.dem,align=self.dem,zoom=self.dem)

        #get outlet point by identifying the highest accumulation value along the perimeter.
        gs.run_command('r.watershed', elevation=self.dem, drainage = self.drain_dir, accumulation = self.accum, flags= 'sabm',memory = self.project.max_mem) ##note that this is in feet
        gs.run_command('v.rast.stats', raster=self.accum, map=f'aoi_{self.select_data}_aoi_raw',method='max',column='accum')
        max_accum = list(gs.parse_command('v.db.select', columns='accum_maximum',map = f'aoi_{self.select_data}_aoi_raw',flags='c').keys())
        for outlet_accum in max_accum:
            gs.run_command('r.mapcalc',expression = f'{self.r_outlet} = if({self.accum} == {outlet_accum},{self.select_data},null())')
            gs.run_command('r.to.vect', input=self.r_outlet, output=self.v_outlet, type='point')
            gs.run_command('v.out.ogr', input=  self.v_outlet ,type = 'point',output = self.outlet, format = 'GeoJSON')
            
        print("Delineating the watersheds")
        gs.run_command('r.stream.basins',direction = self.drain_dir, points = self.v_outlet,basins =self.r_basins, memory = self.project.max_mem)
        gs.run_command('r.stream.extract', elevation=self.dem, accumulation = self.accum, threshold =self.threshold, direction= self.drain_dir,
               stream_raster = self.r_streams,stream_vector = self.v_streams, memory = self.project.max_mem)
        print("Converting the delineated watershed rasters to vectors")
        gs.run_command('r.to.vect', input=self.r_basins, output= self.v_basins, type="area", flags='s')
        gs.run_command('v.out.ogr', input=  self.v_basins ,type = 'area',output = self.basins, format = 'GeoJSON')
        

class RegressionData:
        
    def __init__(self,project, grass_data):
        ## Set variables analysis
        self.project = project
        self.grass_data = grass_data #point of watershed outlet or polygon watershed boundary
    
    def add_col_val(self,grass_layer,col,col_type,val):
        exist_cols = []
        for column in list(gs.parse_command('v.db.connect',map=grass_layer,flags='c')):
            exist_cols.append(column.split('|')[1])
        if col not in exist_cols:
            gs.run_command('v.db.addcolumn',map=grass_layer,columns=f'{col} {col_type}')
        gs.run_command('v.db.update',map=grass_layer,layer=1,column=col,value=int(val))
    
    def set_aoi(self):
        self.grass_data.reset_proj_region()
        gs.run_command('r.mask',raster = self.grass_data.r_basins,overwrite=True)
        gs.run_command('g.region',raster = self.grass_data.dem,align=self.grass_data.dem,zoom=self.grass_data.dem)
    
    def get_raster_avg(self,tif,col):
        gs.run_command('r.import',input=tif,output=col,resample='bilinear',overwrite=True)
        gs.run_command('v.rast.stats', raster=col, map=self.grass_data.v_basins,method='average',column=col,flags='c')
        avg = list(gs.parse_command('v.db.select', columns=f'{col}_average',map = self.grass_data.v_basins,flags='c').keys())
        return avg[0]
        
    def get_drainage_area(self):
        # raster_basin = gs.parse_command('r.info',map=self.grass_data.r_basins,flags='gs')
        # assert self.grass_data.hunits in ['ft','m'], "no units horizontal units set"
        # if self.grass_data.hunits = 'ft':
        #     area = (float(raster_basin['n'])*float(raster_basin['nsres'])*float(raster_basin['ewres']))/(2590136.75519263*np.square(3.28084))
        # else:
        #     area = (float(raster_basin['n'])*float(raster_basin['nsres'])*float(raster_basin['ewres']))/2590136.75519263 #sq m to sq mi
        area_sqmi = float(list(gs.parse_command('v.to.db',map=self.grass_data.v_basins,option='area',flags='p',
                                                    units='miles'))[1].split('|')[1])
        self.drainage_area = area_sqmi
        self.add_col_val(self.grass_data.v_basins,'TDA_SqMi', 'double precision',float(self.drainage_area))
    
    def get_basin_len(self):
        #do shapely stuff
        basin = gpd.read_file(self.grass_data.basins)
        outlet = gpd.read_file(self.grass_data.outlet)
        assert basin.crs == outlet.crs, "Tell curtis he needs to add repojection to this code"
        units_geojson = basin.crs.to_proj4()[basin.crs.to_proj4().find('units')+6:basin.crs.to_proj4().find('+no_defs')-1]
        max_dist = 0
        for point in basin.geometry[0].exterior.coords:
            pnt_dist = shapely.distance(Point(point), outlet.geometry[0])
            if pnt_dist > max_dist:
                max_dist = pnt_dist
                top_coord = point
        if units_geojson == 'us-ft' or units_geojson == 'ft' :
            basin_length = max_dist/5280 #convert feet to miles
        else: #assume meters
            basin_length = max_dist/1609.34 #convert meters to miles
        return basin_length
    
    def get_basin_shape(self):
        perimeter_mi = float(list(gs.parse_command('v.to.db',map=self.grass_data.v_basins,option='perimeter',flags='p',
                                                    units='miles'))[1].split('|')[1])
        basin_length = self.get_basin_len() #miles line drawn between maximum distance along basin perimeter from outlet to minimum distance along basin perimeter to outlet
        basin_width = self.drainage_area/basin_length #miles #Drainage Area / Basin Length
        self.sf = float(self.drainage_area/basin_width)
        self.add_col_val(self.grass_data.v_basins,'SF', 'double precision',float(self.sf))
        self.ii = float((0.5*perimeter_mi*basin_length)/(self.drainage_area + np.square(basin_length)))
        self.add_col_val(self.grass_data.v_basins,'II', 'double precision',float(self.ii))

                              

    def get_basin_relief(self):
        gs.run_command('v.rast.stats', raster=self.grass_data.dem, map=self.grass_data.v_basins,method='max',column='elev')
        max_elevation = list(gs.parse_command('v.db.select', columns='elev_maximum',map = self.grass_data.v_basins,flags='c').keys())
        
        gs.run_command('v.rast.stats', raster=self.grass_data.dem, map=self.grass_data.v_outlet,method='min',column='elev')
        outlet_elevation = list(gs.parse_command('v.db.select', columns='elev_minimum',map = self.grass_data.v_outlet,flags='c').keys())
        assert float(max_elevation) > float(outlet_elevation), 'something is wrong'
        self.br = float(max_elevation) - float(outlet_elevation)
        self.add_col_val(self.grass_data.v_basins,'BR_Ft', 'double precision',float(self.br))
    
    def get_stream_order(self):
        gs.run_command('g.region',raster = self.grass_data.r_streams,align=self.grass_data.r_streams)
        gs.run_command('r.stream.order',stream_rast = self.grass_data.r_streams,direction = self.grass_data.drain_dir,
                       strahler = self.grass_data.r_streams_order, memory=self.project.max_mem)
        stats = gs.parse_command('r.stream.stats',stream_rast = self.grass_data.r_streams_order,direction = self.grass_data.drain_dir,elevation=self.grass_data.dem,
                                 flags='o',memory=self.project.max_mem)
        num_fos = list(stats)[2].split(',')[1]
        self.fos = num_fos
        self.add_col_val(self.grass_data.v_basins,'FOS', 'integer',int(num_fos))
        
    def get_main_ch_slope(self):
        #MCS_FtpMia
        self.grass_data.lfpds = self.grass_data.r_streams +'lfpds'
        self.grass_data.vlfpds = self.grass_data.r_streams +'v_lfpds'
        gs.run_command('r.stream.distance',stream_rast = self.grass_data.r_streams, 
                       direction=self.grass_data.drain_dir,method = 'downstream',
                       distance= self.grass_data.lfpds,flags='o',memory=90000)
        max_dist = gs.parse_command('r.info',map=self.grass_data.lfpds,flags='s')['max']
        gs.run_command('r.mapcalc',expression = f'{self.grass_data.lfpds}_start = if({self.grass_data.lfpds} >= {float(max_dist)-.01},{self.grass_data.aoi},null())')
        gs.run_command('r.to.vect', input=f'{self.grass_data.lfpds}_start', output=f'{self.grass_data.vlfpds}_start', type='point')
        gs.run_command('v.extract',input=f'{self.grass_data.vlfpds}_start',cats='1',output=f'{self.grass_data.vlfpds}_start_op')
        gs.run_command('r.path',input=self.grass_data.drain_dir,format='45degree',start_points=f'{self.grass_data.vlfpds}_start_op',
                       vector_path=self.grass_data.vlfpds)
        gs.run_command('v.segment',input=self.grass_data.vlfpds,rules= self.project.data_dir/'rules.txt',output=f'mcl_p_{self.grass_data.aoi}')
        gs.run_command('v.db.addtable',map=f'mcl_p_{self.grass_data.aoi}')
        gs.run_command('v.rast.stats', raster=self.grass_data.dem, map=f'mcl_p_{self.grass_data.aoi}',method='max',column='elev')
        elevations = list(gs.parse_command('v.db.select', columns='elev_maximum',map = f'mcl_p_{self.grass_data.aoi}',flags='c').keys())
        elevation_diff = float(elevations[0]) - float(elevations[1])
        assert elevation_diff >= 0,'something is wrong'
        #length in miles
        len_mi = (float(list(gs.parse_command('v.to.db',map=self.grass_data.vlfpds,option='length',units='miles',flags='p'))[1].split('|')[1])*0.75)
        ft_mi = elevation_diff / len_mi
        self.mcl_sl = ft_mi
        self.add_col_val(self.grass_data.v_basins,'MCL_Ft_pMi', 'double precision',float(ft_mi))
        
class RegressionEquations:
    def __init__(self,project, grass_data, regression_data,regression_regions:pl.Path):
        ## Set variables analysis
        self.regression_regions = regression_regions
        self.project = project
        self.grass_data = grass_data
        self.regression_data = regression_data #point of watershed outlet or polygon watershed boundary
        self.flows = {}

    def get_regions(self):
        region_coverage = {}
        outlet = gpd.read_file(self.grass_data.outlet)
        basin = gpd.read_file(self.grass_data.basins)
        regions = gpd.read_file(self.regression_regions).to_crs(outlet.crs)
        r = regions['manual_v6'].to_list()
        total_area = basin.iloc[0].geometry.area
        for i in r:
            percent = basin.iloc[0].geometry.intersection(regions.loc[regions['manual_v6'] == i].geometry).iloc[0].area/total_area
            region_coverage[i] = percent
        self.regional_coverage = region_coverage
    
    def apply_equations(self,region,interval):
        CDA_SqMi = float(self.regression_data.drainage_area)
        PRISM_yr_mm = float(self.regression_data.PRISMyr_mm)
        MCS_FtpMi = float(self.regression_data.mcl_sl)
        FOS = int(self.regression_data.fos)
        SF = float(self.regression_data.sf)
        II = float(self.regression_data.ii)
        CN = int(float(self.regression_data.CN_mean))
        eqns = {1:
                {50: 10**-15.54*CDA_SqMi**0.33*PRISM_yr_mm**6.10*MCS_FtpMi**0.67,
                 20: 10**-18.31*CDA_SqMi**0.47*PRISM_yr_mm**7.07* MCS_FtpMi**0.92, 
                 10: 10**-19.42*CDA_SqMi**0.55*PRISM_yr_mm**7.44* MCS_FtpMi**1.07,
                 4: 10**-20.34*CDA_SqMi**0.64*PRISM_yr_mm**7.72* MCS_FtpMi**1.23,
                 2: 10**-20.80*CDA_SqMi**0.70*PRISM_yr_mm**7.86* MCS_FtpMi**1.34,
                 1: 10**-21.13*CDA_SqMi**0.75*PRISM_yr_mm**7.94* MCS_FtpMi**1.44,
                 0.5: 10**-21.35*CDA_SqMi**0.80*PRISM_yr_mm**7.99* MCS_FtpMi**1.54,
                 0.2: 10**-21.53*CDA_SqMi**0.86*PRISM_yr_mm**8.01* MCS_FtpMi**1.66},
                2:
                {50: 10**-15.54*CDA_SqMi**0.33*PRISM_yr_mm**6.10*MCS_FtpMi**0.67,
                 20: 10**-18.31*CDA_SqMi**0.47*PRISM_yr_mm**7.07* MCS_FtpMi**0.92, 
                 10: 10**-19.42*CDA_SqMi**0.55*PRISM_yr_mm**7.44* MCS_FtpMi**1.07,
                 4: 10**-20.34*CDA_SqMi**0.64*PRISM_yr_mm**7.72* MCS_FtpMi**1.23,
                 2: 10**-20.80*CDA_SqMi**0.70*PRISM_yr_mm**7.86* MCS_FtpMi**1.34,
                 1: 10**-21.13*CDA_SqMi**0.75*PRISM_yr_mm**7.94* MCS_FtpMi**1.44,
                 0.5: 10**-21.35*CDA_SqMi**0.80*PRISM_yr_mm**7.99* MCS_FtpMi**1.54,
                 0.2: 10**-21.53*CDA_SqMi**0.86*PRISM_yr_mm**8.01* MCS_FtpMi**1.66},
                3:
                {50: 10**-15.54*CDA_SqMi**0.33*PRISM_yr_mm**6.10*MCS_FtpMi**0.67,
                 20: 10**-18.31*CDA_SqMi**0.47*PRISM_yr_mm**7.07* MCS_FtpMi**0.92, 
                 10: 10**-19.42*CDA_SqMi**0.55*PRISM_yr_mm**7.44* MCS_FtpMi**1.07,
                 4: 10**-20.34*CDA_SqMi**0.64*PRISM_yr_mm**7.72* MCS_FtpMi**1.23,
                 2: 10**-20.80*CDA_SqMi**0.70*PRISM_yr_mm**7.86* MCS_FtpMi**1.34,
                 1: 10**-21.13*CDA_SqMi**0.75*PRISM_yr_mm**7.94* MCS_FtpMi**1.44,
                 0.5: 10**-21.35*CDA_SqMi**0.80*PRISM_yr_mm**7.99* MCS_FtpMi**1.54,
                 0.2: 10**-21.53*CDA_SqMi**0.86*PRISM_yr_mm**8.01* MCS_FtpMi**1.66},
                4:
                {50: 10**-15.54*CDA_SqMi**0.33*PRISM_yr_mm**6.10*MCS_FtpMi**0.67,
                 20: 10**-18.31*CDA_SqMi**0.47*PRISM_yr_mm**7.07* MCS_FtpMi**0.92, 
                 10: 10**-19.42*CDA_SqMi**0.55*PRISM_yr_mm**7.44* MCS_FtpMi**1.07,
                 4: 10**-20.34*CDA_SqMi**0.64*PRISM_yr_mm**7.72* MCS_FtpMi**1.23,
                 2: 10**-20.80*CDA_SqMi**0.70*PRISM_yr_mm**7.86* MCS_FtpMi**1.34,
                 1: 10**-21.13*CDA_SqMi**0.75*PRISM_yr_mm**7.94* MCS_FtpMi**1.44,
                 0.5: 10**-21.35*CDA_SqMi**0.80*PRISM_yr_mm**7.99* MCS_FtpMi**1.54,
                 0.2: 10**-21.53*CDA_SqMi**0.86*PRISM_yr_mm**8.01* MCS_FtpMi**1.66},
                5:
                {50: 10**-15.54*CDA_SqMi**0.33*PRISM_yr_mm**6.10*MCS_FtpMi**0.67,
                 20: 10**-18.31*CDA_SqMi**0.47*PRISM_yr_mm**7.07* MCS_FtpMi**0.92, 
                 10: 10**-19.42*CDA_SqMi**0.55*PRISM_yr_mm**7.44* MCS_FtpMi**1.07,
                 4: 10**-20.34*CDA_SqMi**0.64*PRISM_yr_mm**7.72* MCS_FtpMi**1.23,
                 2: 10**-20.80*CDA_SqMi**0.70*PRISM_yr_mm**7.86* MCS_FtpMi**1.34,
                 1: 10**-21.13*CDA_SqMi**0.75*PRISM_yr_mm**7.94* MCS_FtpMi**1.44,
                 0.5: 10**-21.35*CDA_SqMi**0.80*PRISM_yr_mm**7.99* MCS_FtpMi**1.54,
                 0.2: 10**-21.53*CDA_SqMi**0.86*PRISM_yr_mm**8.01* MCS_FtpMi**1.66},
               }
        flow = eqns[region][interval]
        return flow
        
    def calc_flows(self):
        tot_pct = 1
        reccurrence = [50,20,10,4,2,1,0.5,0.2]
        for interval in reccurrence:
            flow = 0
            for region, percent in self.regional_coverage.items():
                flow += self.apply_equations(region,interval)*percent
            self.flows[interval] = flow
                    
                
                
        