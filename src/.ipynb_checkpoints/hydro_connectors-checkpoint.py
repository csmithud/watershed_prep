def adjust_nhd_lines_no_snap(dem,v_stream,outputs_dir,overwrite = False):
    print("combining watershed lines and nhd lines")
    gs.run_command("v.build.polylines",input=v_stream,output= '{}_poly'.format(v_stream),cats='same')
    gs.run_command("v.to.points", input= '{}_poly'.format(v_stream), type='line', output='{}_pts'.format(v_stream),dmax=5)
    gs.run_command('v.drape',input='{}_pts'.format(v_stream),elevation=dem,type='point',output='{}_pts_3d'.format(v_stream))
    
def adjust_nhd_lines(dem,v_stream,outputs_dir,overwrite = False):
    logger.info("combining watershed lines and nhd lines")
    #get all HUC12s within select data
    huc12_dict = gs.parse_command('v.db.select',map=huc12, flags='v')
    hucs = []
    GRASS_vector_files = [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
    if v_stream+'_buffer' not in GRASS_vector_files or force:
        gs.run_command('v.buffer',input=v_stream,output=v_stream+'_buffer',distance = 50) #buffer stream corridors by 50 meters
    for keys in list(huc12_dict.keys()):
        key, val =  keys.split('|')
        if val.find(select_data)>=0:
            hucs.append(val)
    #for testing
    hucs = [hucs[0]]
    for huc in hucs:
        v_huc12 = f'huc12_{huc}'
        v_stream_huc_buff = v_stream+'_buffer_'+huc
        v_stream_huc = v_stream+'_'+huc
        #Extract NHD flow lines for upstream HUC12 areas
        gs.run_command('v.extract', input =huc12, where= "%s = '%s'" % ("HUC12",huc), output = v_huc12)
        gs.run_command('v.overlay', ainput = v_huc12,binput = v_stream+'_buffer', output = v_stream_huc_buff, operator = 'and')
        #intersect 
        gs.run_command('v.select',ainput=v_stream,atype='line',binput=v_huc12,btype='area',output=v_stream_huc,operator='intersects')
        #mask dem for faster processing
        gs.run_command('r.mask',vector = v_stream_huc_buff)
        #snap points to accum lines for more accurate streams
        reg = gs.parse_command('g.region',raster=dem, align=dem, zoom= dem,flags='pgm')
        cell_size = (float(reg.nsres)+float(reg.nsres))/2
        max_mem = get_RAM()
        gs.run_command('r.watershed', elevation= dem, threshold=10000 // cell_size**2, stream='r_'+v_stream_huc+'_stream_grass', flags='sab', overwrite=overwrite, memory = max_mem)
        #reclassify streams as 
        #gs.run_command('r.stream.extract', elevation=dem, threshold =100, stream_raster = 'r_'+v_stream+'_stream_grass',stream_vector = v_stream+'_grass',memory = max_mem)
        gs.run_command("v.build.polylines",input=v_stream_huc,output= '{}_poly'.format(v_stream_huc),cats='same')
        radius = 50 #meters
        gs.run_command("v.to.points", input= '{}_poly'.format(v_stream_huc), type='line', output='{}_pts'.format(v_stream_huc),dmax=5)
        gs.run_command("r.stream.snap", input= '{}_pts'.format(v_stream_huc),stream_rast='r_'+v_stream_huc+'_stream_grass',
                           output ='{0}_pts_snapped'.format(v_stream_huc),
                          radius=radius, memory = max_mem)
        gs.run_command('v.db.connect',map='{}_pts_snapped'.format(v_stream_huc),table='{}_pts'.format(v_stream_huc))
        logger.info("capturing elevations along nhd lines")
        gs.run_command('v.drape',input='{}_pts_snapped'.format(v_stream_huc),elevation=dem,type='point',output='{}_pts_3d'.format(v_stream_huc))
        
        gs.run_command('r.mask',flags='r')
        
def roads_detector(df_pts,ratio):
    #roads and their average width in USA is 9 - 15 m (Hughes et al.,2004). Reccomends median filter of 21m
    #assume 2 foot max delta between min and max per of rise per 10 meters of lenth -
    #is road if len / (maxZ - minZ) < 5, else not 
    distance = df_pts['distance'].sum()
    if distance <= 50 and distance >= 5:
        height = df_pts['z'].max() - df_pts['z'].min()
        if distance / height <= ratio:
            return df_pts, distance / height
        else:
            return pd.DataFrame(columns = df_pts.columns.to_list()), 0
    else:
        return pd.DataFrame(columns = df_pts.columns.to_list()), 0
    
def create_hydro_connectors(dem,v_stream,outputs_dir, projection, ratio = 15, overwrite = False):
    hydro_connectors = dem[:dem.find('dem')]+'hydro_connectors'
    GRASS_vector_files= [file for line in gs.list_strings(type='vector') for file in [line.split("@")[0]]]
    if  hydro_connectors in GRASS_vector_files and overwrite is False:
        logger.info("hydroconnectors exist. Not recreating.")
    else:
        #clean up existing files
        if os.path.exists(outputs_dir) and os.path.isdir(outputs_dir):
            shutil.rmtree(outputs_dir)
        os.makedirs(outputs_dir)
        gs.run_command('g.remove', type ='vector', name=hydro_connectors, flags ='f')

        logger.info("exporting nhd lines for road and dam analysis")

        cats = list(gs.parse_command('v.db.select',map='{}_pts_3d'.format(v_stream),columns='cat',flags='c',separator=',').keys())
        for cat in cats:
            gs.run_command('v.out.ascii',flags='c',type='point',input ='{}_pts_3d'.format(v_stream), cats= cat, output=outputs_dir/'nhd_out_{}.txt'.format(cat),format='wkt')
            
            points = []
            df_points = pd.DataFrame()
            with open(outputs_dir/"nhd_out_{}.txt".format(cat), 'r') as fd:
                txt = fd.read()
            points = txt.splitlines()
            df_points = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(points, index=None, crs=int(projection)))    
            temp = get_increases_along_streams(df_points.geometry.to_list(),cat,outputs_dir,int(projection),ratio=ratio)
            
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
                road_pts, slope = roads_detector(df_seg,ratio)
                if road_pts.empty is False:
                    road_pts = gpd.GeoDataFrame(road_pts, geometry=road_pts.apply(lambda r: shapely.geometry.Point(r.x,r.y,r.z), axis=1))
                    road_pts['segment'] = seg
                    seg+=1
                    # generate Linestrings grouping by station
                    gls = gpd.GeoDataFrame(geometry=gpd.GeoSeries(road_pts.groupby('segment').apply(lambda d: shapely.geometry.LineString(d["geometry"].values))))
                    gls['ratio'] = slope
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
        
def topo_carve(dem,overwrite = False):
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
    return output