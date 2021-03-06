"""
Build Thermal Maps and LSOA-aggregated Residential Heat Loss vector

Inputs:
- Counties table containing polygon for the chosen county
- LSOA table for the chosen county containing LSOA polygons - geojson
- County name
- Land Cover map

Outputs:
- Table of available scenes, cloud cover, proc level, collection, category ...
- Images for visual inspection of cloud masks
- Images for visual inspection of land cover mask
- Time-series aggregated thermal maps 
- GeoJSON vector of LSOA-aggregated RHL (various methods)

Procedure:
- Select county, read shape, get bounds for AoI
- Define path,row of interest (use lookup table)
- Select Season(s) of Interest (SoI) given by date range(s)
- Search landsat data on Google Cloud for scenes meeting criteria
- Open and read windows
- Call functions to do cloud masking
- Stack masked images
- Call functions to perform Ambient Temp Correction
- Open and read Land Cover map
- Perform land cover masking and aggregate over LSOAs

Usage:
import builder
params = (date_label,place_label,dates_of_interest,pathrows,max_cloud,cloud_mask_bits)
# See bottom of module for how to set up these arguments
builder.main(params)


Dependencies:
land_surface_temperature
landsat
met_climate
common/raster_utils
common/geoplot
config

This is all based on the original procedure in LSOA_LST_builder.ipynb
"""


import os
import glob
from collections import OrderedDict
from urllib.request import urlopen
import pprint as pp
import numpy as np
from numpy import ma as ma
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.plot as rplt
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterstats import zonal_stats
from scipy.ndimage import convolve, filters

import landsat
import land_surface_temperature as lst
import met_climate
from common import raster_utils as ru
from common import geoplot as gpl
 
# GLOBALS defined here
import config as cf  
rootdir = cf.rootdir
tirband = cf.tirband
qaband = cf.qaband 
bband = cf.bband
gband = cf.gband
rband = cf.rband
nband = cf.nband

def parse_metadata(lines):
    objects = {}
    cur_object = objects
    for line in lines:
        line = line.decode('ascii').strip()
        if line == 'END':
            continue
        tag, val = line.split('=')
        tag = tag.rstrip()
        val = val.lstrip()
        if tag == 'GROUP':
            #print('Created Group '+val)
            cur_object[val] = {}
            pre_object = cur_object
            cur_object = cur_object[val]
            # Add the next few lines to this sub-dict
        elif tag == 'END_GROUP':
            #print('Exiting Group '+val)
            cur_object = pre_object
            # Go back to the parent dict level
        else:
            #print('Add object '+tag)
            cur_object[tag] = val.strip('"')

    return objects

    # see also http://community.hexagongeospatial.com/t5/Spatial-Modeler-Tutorials/Using-the-Python-Script-operator-to-ingest-sensor-metadata-2016/ta-p/6233

def get_metadata(scene_urls):
    """Read metadata of scenes 
    """

    meta_list = []
    for url in scene_urls:
        with urlopen(url) as f:
            m = f.readlines()
            f.close()
        
        meta = parse_metadata(m)
        meta_list += [meta['L1_METADATA_FILE']]

    return meta_list


def display_qamask(scene_url,output_plot_dir,cloud_mask_bits,**aoi_kwargs):

    filename = output_plot_dir + \
                scene_url.split('/')[-1].replace(
                    '.TIF','_mask_check.png'
                    )
    
    legend = ('Legend:' ,
              'Magenta = terrain occlusion',
              'white = cloud bit',
              'Red = cloud conf med/high',
              'Cyan = cirrus conf med/high',
              'Blue = cloud shadow conf med/high',
              'Green = snow/ice conf med/high'
              )

    scene_tir = scene_url
    scene_bqa = scene_url.replace('B'+tirband,'B'+qaband)

    with rasterio.open(scene_bqa) as bqa:
        with rasterio.open(scene_tir) as tir:

            bqa_data,bqa_trans = ru.read_in_aoi(bqa,**aoi_kwargs)
            tir_data,tir_trans = ru.read_in_aoi(tir,**aoi_kwargs)
            
    bqa_data = bqa_data[0,:,:]
    tir_data = tir_data[0,:,:]
    tir_data = ma.array(tir_data,dtype=float,
                        mask=ru.mask_qa(bqa_data,bitmask=0b1))

    (ymin,ymax) = (0, tir_data.shape[0])
    (xmin,xmax) = (0, tir_data.shape[1])
    
    # Plot unmasked data
    fig,axes = gpl.make_figure(shape=2,figsize=(40,20))
    ax1,norm1 = gpl.raster(
        axes[0],
        tir_data[ymin:ymax,xmin:xmax],
        rescale_kind='hist_eq',
        colormap='Greys_r',
        title='TIR greyscale with masks')

    # Make mask arrays
    smw = 11
           
    mask_occ_sm = filters.maximum_filter(ru.mask_qa(bqa_data,bits=[1]),size=smw)
    mask_cloud_sm = filters.maximum_filter(ru.mask_qa(bqa_data,bits=[0,4]),size=smw)
    mask_clcon_sm = filters.maximum_filter(ru.mask_qa(bqa_data,bits=[6]),size=smw)
    mask_cicon_sm = filters.maximum_filter(ru.mask_qa(bqa_data,bits=[12]),size=smw)
    mask_cscon_sm = filters.maximum_filter(ru.mask_qa(bqa_data,bits=[8]),size=smw)
    mask_sncon_sm = filters.maximum_filter(ru.mask_qa(bqa_data,bits=[10]),size=smw)   

    # Filled contours for the various "confidence" masks
    ax1.contourf(mask_occ_sm[ymin:ymax,xmin:xmax],levels=[0.5,1],
                   colors='magenta',antialiased=True)
    ax1.contourf(mask_sncon_sm[ymin:ymax,xmin:xmax],[0.5,1],
                   colors='green',antialiased=True)
    ax1.contourf(mask_cscon_sm[ymin:ymax,xmin:xmax],[0.5,1],
                   colors='blue',antialiased=True)
    ax1.contourf(mask_clcon_sm[ymin:ymax,xmin:xmax],[0.5,1],
                   colors='red',antialiased=True)
    ax1.contourf(mask_cicon_sm[ymin:ymax,xmin:xmax],[0.5,1],
                   colors='cyan',antialiased=True)
    
    # Unfilled contour for the simple cloud bit
    ax1.contour(mask_cloud_sm[ymin:ymax,xmin:xmax],levels=[0.5],
                   colors='white',linewidths=0.5,antialiased=True)

    # Combined mask of selected bits
    mask_all = filters.maximum_filter(
        ru.mask_qa(bqa_data,bits=cloud_mask_bits),
        size=smw
        )

    tir_data_mask_all = ma.array(tir_data,
                                 mask=mask_all,
                                 fill_value=0
                                 ).filled()

    # Plot masked data
    ax2,norm2 = gpl.raster(
        axes[1],
        tir_data_mask_all[ymin:ymax,xmin:xmax],
        rescale_kind='hist_eq',
        colormap='hot',
        title='Masked TIR',
        titlesize='xx-large')
    
    # Add some text and save
    ax1.text(1,1,'\n'.join(legend),
             transform=ax1.transAxes,
             clip_on=False)
    fig.suptitle('{} smw={}'.format(scene_url.split('/')[-1], smw),
                 fontsize='xx-large')
    fig.savefig(filename)

    return

def display_rgb(scene_url,output_plot_dir,**aoi_kwargs):
    
    filename = output_plot_dir + \
                scene_url.split('/')[-1].replace(
                    f'B{tirband}.TIF','RGB.png'
                    )
    scene_b = scene_url.replace('B'+tirband,'B'+bband)
    scene_g = scene_url.replace('B'+tirband,'B'+gband)
    scene_r = scene_url.replace('B'+tirband,'B'+rband)
    scene_bqa = scene_url.replace('B'+tirband,'B'+qaband)

    with rasterio.open(scene_b) as src:
        blue_data,blue_trans = ru.read_in_aoi(src,**aoi_kwargs)
    with rasterio.open(scene_g) as src:
        green_data,green_trans = ru.read_in_aoi(src,**aoi_kwargs)
    with rasterio.open(scene_r) as src:
        red_data,red_trans = ru.read_in_aoi(src,**aoi_kwargs)
    with rasterio.open(scene_bqa) as src:
        bqa_data,bqa_trans = ru.read_in_aoi(src,**aoi_kwargs)
    
    bqa_data = bqa_data.squeeze()
    for arr in (blue_data,green_data,red_data):
        arr = ma.array(arr,dtype=float,
                       mask=ru.mask_qa(bqa_data,bits=[0]),
                       fill_value=0.
                       ).filled()   
    blue_data = blue_data.squeeze()
    green_data = green_data.squeeze()
    red_data = red_data.squeeze()
    
    rgb_data = np.array(np.dstack([red_data,green_data,blue_data]),dtype=float)
    
    (ymin,ymax) = (0, rgb_data.shape[0])
    (xmin,xmax) = (0, rgb_data.shape[1])
                        
    # Plot RGB data
    fig,ax1 = gpl.make_figure(figsize=(20,20))
    
    ax1,norm1 = gpl.raster(
        ax1,
        rgb_data[ymin:ymax,xmin:xmax,:],
        rescale_kind='clahe',
        use_rst_plot=False
        )

    fig.savefig(filename)

    return

def get_ceda_password():
    f = open(rootdir+'uk_data/ceda_password.txt')
    content = f.readline().strip()
    f.close()
    return content

def stack_tir(scene_urls,cloud_mask_bits,aoi,aoi_crs,
              subtract_median_lst=True,subtract_air_temp=False):
    """
    Convert clipped and masked TIR scenes to LST, then aggregate 
    Also allow for calculation of xLST: subtracting mean air temp 
    on day of acquisition, and aggregating the result

    This has to be re-run each time you want to produce xLST,rxLST,rLST stacks

    """
    if subtract_air_temp:
        ceda_password = get_ceda_password()
        at = met_climate.access_ukcp09(cf.ceda_username,ceda_password)

    
    # with rasterio.open(scene_bqa) as bqa:
    #     with rasterio.open(scene_tir) as tir:

    #         bqa_data,bqa_trans = ru.read_in_aoi(bqa,**aoi_kwargs)
    #         tir_data,tir_trans = ru.read_in_aoi(tir,**aoi_kwargs)
            
    # bqa_data = bqa_data[0,:,:]
    # tir_data = tir_data[0,:,:]
    # tir_data = ma.array(tir_data,dtype=float,
    #                     mask=ru.mask_qa(bqa_data,bitmask=0b1))

    # (ymin,ymax) = (0, tir_data.shape[0])
    # (xmin,xmax) = (0, tir_data.shape[1])
        
    counter=-1
    for scene_url in scene_urls:
        counter+=1
        scene_tir = scene_url
        scene_bqa = scene_url.replace('B'+tirband,'B'+qaband)
        scene_red = scene_url.replace('B'+tirband,'B'+rband)
        scene_nir = scene_url.replace('B'+tirband,'B'+nband)
        scene_metadata = scene_url.replace('B'+tirband+'.TIF','MTL.txt')

        print('Reading scene {}'.format(counter+1))
        try:
            with rasterio.open(scene_bqa) as bqa:
                #print(scene_bqa)
                bqa_data,bqa_trans = ru.read_in_aoi(bqa,aoi=aoi,aoi_crs=aoi_crs)

            with rasterio.open(scene_tir) as tir:
                #print(scene_tir)
                tir_data,tir_trans = ru.read_in_aoi(tir,aoi=aoi,aoi_crs=aoi_crs)
                tir_crs = tir.crs
                tir_profile = tir.profile

            with rasterio.open(scene_red) as red:
                #print(scene_red)
                red_data,red_trans = ru.read_in_aoi(red,aoi=aoi,aoi_crs=aoi_crs)
                red_crs = red.crs

            with rasterio.open(scene_nir) as nir:
                #print(scene_nir)
                nir_data,nir_trans = ru.read_in_aoi(nir,aoi=aoi,aoi_crs=aoi_crs)
        
        except OSError as e:
            print('ERROR',e)
            print('skipping scene')
            counter = counter-1
            continue
        
        # Determine size of stack allowing for AoI to extend outside of scene
        if counter == 0:
            aoi_box = rasterio.warp.transform_bounds(aoi_crs,tir_crs,*aoi.values())
            aoi_left, aoi_bottom, aoi_right, aoi_top = aoi_box
            aoi_box = dict(zip(('minx','miny','maxx','maxy'),aoi_box))
            # rowmin,colmin = (bqa.index(aoi_left,aoi_top)) #,op=round))
            # rowmax,colmax = (bqa.index(aoi_right,aoi_bottom)) #,op=round))
            # The above two lines are fine but the following does not 
            # require the rasterio dataset to be kept open
            rowmin,colmin = rasterio.transform.rowcol(tir_trans,aoi_left,aoi_top)
            rowmax,colmax = rasterio.transform.rowcol(tir_trans,aoi_right,aoi_bottom)
            stack_height,stack_width = (rowmax-rowmin,colmax-colmin)
            lst_stack = (ma.zeros((len(scene_urls),stack_height,stack_width),
                                  dtype=np.float,fill_value=np.nan
                                 )+np.nan)   
            
        # Determine size of intersect in THIS scene
        intersect = ru.aoi_scene_intersection(aoi_box,bqa)
        ins_left, ins_bottom, ins_right, ins_top = intersect.bounds
        #rowmin,colmin = (bqa.index(ins_left,ins_top,op=round))
        #rowmax,colmax = (bqa.index(ins_right,ins_bottom,op=round))
        # The above two lines are incorrect now that we read a window:
        # We need to transform the coordinates into the row,col of 
        # the window, not the original file.
        rowmin,colmin = rasterio.transform.rowcol(tir_trans,ins_left,ins_top)
        rowmax,colmax = rasterio.transform.rowcol(tir_trans,ins_right,ins_bottom)

        try:
            # Subset data 
            bqa_data = ma.array(bqa_data[0,rowmin:rowmax,colmin:colmax])
            tir_data = ma.array(tir_data[0,rowmin:rowmax,colmin:colmax])
            red_data = ma.array(red_data[0,rowmin:rowmax,colmin:colmax])
            nir_data = ma.array(nir_data[0,rowmin:rowmax,colmin:colmax])
            assert tir_data.shape == lst_stack.shape[1:]
        except (IndexError,AssertionError) as e:
            print('ERROR:',e)
            print('loop count',counter)
            print(tir_data.shape, lst_stack.shape)
            print(rowmin,rowmax,colmin,colmax)
            import pdb; pdb.set_trace()

        lst_data = lst.calculate_land_surface_temperature_NB(
                        red_data, nir_data, tir_data,
                        red_trans, tir_trans, 
                        red_crs, tir_crs, scene_metadata
                        )
        
        # Masks
        smw = 11
        mask_all = filters.maximum_filter(
                            ru.mask_qa(bqa_data,bits=cloud_mask_bits),size=smw
                            )

        lst_data_mask_all = ma.array(lst_data,
                            mask=mask_all,
                            dtype=np.float,
                            fill_value=np.nan) #.filled()

        # After masking, reproject
        # not necessary if they share a CRS
        if counter > 0:
            assert tir_crs == prev_crs
        prev_crs = tir_crs

        # Now do some normalisation
        if subtract_air_temp:
            filename = scene_tir.split('/')[-1]
            datestring = filename.split('_')[3]

            atscene = met_climate.dummy_scene( 
                tir_crs, tir_trans, aoi_box,(stack_height,stack_width))

            # import pdb; pdb.set_trace()
            # If the following fails, it may mean there was a problem setting up the session
            atdata = at.grid_temp_over_scene(
                atscene, datestring, interpolation='linear')
            atdata = atdata[rowmin:rowmax,colmin:colmax]
            assert lst_data_mask_all.shape == atdata.shape
            lst_data_mask_all = ma.array(
                            lst_data_mask_all - atdata,
                            mask=mask_all,
                            fill_value=np.nan)
            
            if subtract_median_lst:
                # ALSO subtract median xLST
                medval = ma.median(lst_data_mask_all)
                lst_data_mask_all = ma.array(
                            lst_data_mask_all - medval,
                            mask=mask_all,
                            fill_value=np.nan)
                
        elif subtract_median_lst:
            # Subtract median LST from scene (within QA mask) 
        
            medval = ma.median(lst_data_mask_all)
            lst_data_mask_all = ma.array(
                        lst_data_mask_all - medval,
                        mask=mask_all,
                        fill_value=np.nan)
        
        # Then add to stack
        lst_stack[counter,:,:] = lst_data_mask_all

    # Make profile for file output
    N_layers = counter+1
    tir_profile.update(
        dtype=rasterio.float64,
        width=stack_width,
        height=stack_height,
        transform=tir_trans,
        count=N_layers,
        compress='lzw'
        )


    return lst_stack, tir_profile

def subtract_mean_at(meanlstfile,dates_of_interest):
    """With existing 'rLST' product, subtract mean AT over range of months
    
    Inputs: 
    meanlstfile = where to find the rLST_mean data
    dates_of_interest = list of dates enclosing seasons of interest 
        eg [['20101101','20110228']]
        These must be strings formatted as YYYYMMDD 
        and the list must have two dimensions

    Return: 
    lst_mean = mean xrLST = rLST - mean(AT)
    profile = rasterio profile of meanlstfile
    """
    ceda_password = get_ceda_password()
    at = met_climate.access_ukcp09(cf.ceda_username,ceda_password)

    # work out which months are within the dates of interest
    first_month = int(max([
        dates_of_interest[a][0][4:6] 
        for a in range(np.shape(dates_of_interest)[0])
    ]))
    last_month = int(max([
        dates_of_interest[a][1][4:6] 
        for a in range(np.shape(dates_of_interest)[0])
    ]))
    if first_month>last_month:
        months = list(range(first_month,13)) + list(range(1,last_month+1))
    else:
        months = list(range(first_month,last_month+1))

    # get the LST data that we are to subtract the AT from
    with rasterio.open(meanlstfile) as lstdata: 
        profile = lstdata.profile
        lst_mean = lstdata.read()

        at_stack = np.zeros([len(months),lst_mean.shape[1],lst_mean.shape[2]])

        # Get mean AT in each month
        for ii in range(len(months)):
            month = months[ii]
            atdata = at.grid_temp_over_scene(
                lstdata, month, interpolation='linear')
            try:
                assert at_stack[ii,:,:].shape == atdata.shape
            except:
                import pdb; pdb.set_trace()
            at_stack[ii,:,:] = atdata

    # average over the months
    at_mean = at_stack.mean(axis=0).reshape([1,at_stack.shape[1],at_stack.shape[2]])
    try:
        assert lst_mean.shape == at_mean.shape 
    except:
        import pdb; pdb.set_trace()

    xrlst_mean = ma.array(lst_mean - at_mean,
                          mask=lst_mean==np.nan, # ==0
                          fill_value=np.nan).filled()
    return xrlst_mean, profile

def main(*args,diagnostics=False):
    """Main function
    
    Input:
    par = (                 # E.g.
        place_label,        # 'derbyshire'
        date_label,         # '2014-2016'
        dates_of_interest,  # [['20141101','20150228'],['20151101','20160228']]
        pathrows            # [['202','023'],['203','023']]
        max_cloud           # 70
        cloud_mask_bits     # [0,1,4,8,10]
        )
    diagnostics = whether to make images showing the QA masks and RGB images 
        of each scene

    """
    date_label,place_label,dates_of_interest,pathrows,max_cloud,cloud_mask_bits = args
    assert type(place_label)==str and len(place_label)>0
    assert type(date_label)==str and len(date_label)>0
    assert len(dates_of_interest)>0
    assert len(pathrows)>0

    output_plot_dir = rootdir+'output_LSOA_LST/{}_{}/'.format(
        place_label,date_label)
    if len(glob.glob(output_plot_dir))==0:
        os.system('mkdir -p {}'.format(output_plot_dir))
    else:
        print('Warning: output directory exists and will be overwritten: ' 
                +'{}'.format(output_plot_dir))


    """
    Define the AoI based on County data
    """
    county_string = place_label+'*'
    counties = gpd.read_file(cf.uk_counties_shapefile) 
    county_ind = np.where(counties['ctyua15nm'].str.match(
        county_string,case=False,na=False))[:][0]
    print('County in shapefile: ',county_ind,
            counties.iloc[county_ind]['ctyua15nm'].values)

    county_bounds = counties.bounds.iloc[county_ind[0]].to_dict()
    county_crs = counties.crs
            
    """
    Query Google Cloud for Landsat scenes
    """
    scenes, scene_dates = [],[]
    scenebucket = landsat.gcs()
    print('Searching for Landsat scenes in {}'.format(scenebucket.bucket_name))     
    for SoI_counter in range(np.shape(dates_of_interest)[0]):
        time_interval = (pd.to_datetime(dates_of_interest[SoI_counter][0]),
                         pd.to_datetime(dates_of_interest[SoI_counter][1]))
        print('In {}:'.format(dates_of_interest[SoI_counter]))
        
        for l8_path,l8_row in pathrows:
            myscenes,mydates = scenebucket.find_in_bucket(
                l8_path,l8_row,tirband,Collection=True,
                minDate=time_interval[0],
                maxDate=time_interval[1],
                UniqueAcq=True
                )
            print('P,R {},{}: {} scenes found'.format(
                l8_path,l8_row,len(myscenes)))
            
            scenes += myscenes
            scene_dates += mydates
            # break
        # break 

    sort = np.argsort(scene_dates)
    scene_urls = np.array([scenebucket.https_stub + s for s in np.array(scenes)[sort]])
    scene_dates = np.array(scene_dates)[sort]

    print('Total {} scenes found'.format(len(scene_urls)))

    meta_urls = np.array([s.replace('B10.TIF','MTL.txt') for s in scene_urls])

    """
    Get Metadata for scenes and store in neat table
    """
    print('Read metadata of scenes')
    meta_list = np.array(get_metadata(meta_urls))

    scenes_table = pd.DataFrame.from_dict(OrderedDict({
            'Scene':     [m['METADATA_FILE_INFO']['LANDSAT_PRODUCT_ID'] 
                          for m in meta_list],
            'AcqDate':   [m['PRODUCT_METADATA']['DATE_ACQUIRED']
                          for m in meta_list],
            'CloudLand': [float(m['IMAGE_ATTRIBUTES']['CLOUD_COVER_LAND'])
                          for m in meta_list],
            'DataType':  [m['PRODUCT_METADATA']['DATA_TYPE']
                          for m in meta_list],
            'Category':  [m['PRODUCT_METADATA']['COLLECTION_CATEGORY']
                          for m in meta_list],
            }))

    """
    Filter out anything with too much cloud cover
    """
    good_cloud, = np.where(scenes_table['CloudLand'].values <= max_cloud)

    scene_urls = scene_urls[good_cloud]
    scene_dates = scene_dates[good_cloud]
    meta_list = meta_list[good_cloud]
    scenes_table = scenes_table.iloc[good_cloud]
    
    print('Found {} scenes with land cloud cover <= {}%'.format(
            len(scene_urls),max_cloud))

    """
    Output table of what we're going to use
    """
    scenes_table.to_csv(output_plot_dir+'scenes_list.dat',sep=' ')

    """
    Make RGB images
    """
    if diagnostics:
        print('Making RGB images')
        for scene_url in scene_urls:
            display_rgb(scene_url,output_plot_dir,
                        aoi=county_bounds,aoi_crs=county_crs)

    """
    Make QA images
    """
    if diagnostics:
        print('Making QA/TIR images')
        for scene_url in scene_urls:
            display_qamask(scene_url,output_plot_dir,
                           aoi=county_bounds,aoi_crs=county_crs)

    """
    QA-mask and Aggregate time-series of LST
    """

    #counties = counties.to_crs(scene.crs)
    print('Stacking TIR data...')
    
    # check that 'rLST' is produced before 'xrLST'
    products = cf.products
    if ('xrLST' in products) and not ('rLST' in products):
        products = ['rLST'] + products
    elif (np.flatnonzero('xrLST' == np.array(products))[0]
          < np.flatnonzero('rLST' == np.array(products))[0]):
        products = (['rLST'] 
            + list(np.array(products)[
                np.flatnonzero('rLST' != np.array(products))]
            )
        )

    for product_name in products:

        print('Producing {}'.format(product_name))
        
        if product_name!='xrLST':
            lst_stack,profile = stack_tir(scene_urls,cloud_mask_bits,
                                   county_bounds,county_crs,
                                   subtract_median_lst=('r' in product_name),
                                   subtract_air_temp=('x' in product_name)
                                   )

            lst_count = lst_stack.count(axis=0)
            lst_mean = ma.array(lst_stack.mean(axis=0),
                                mask=~(lst_count>0),
                                fill_value=np.nan
                                ).filled()
            lst_var = ma.array(lst_stack.var(axis=0),
                                mask=~(lst_count>0),
                                fill_value=np.nan
                                ).filled()
        
        else:
            meanlstfile = output_plot_dir+'rLST_mean.tif'
            lst_mean,profile = subtract_mean_at(meanlstfile,dates_of_interest)
            
        """
        Output the stack data
        """
        print('Outputting stacked heat maps as raster')
        # import pdb; pdb.set_trace()
            
        with rasterio.Env():
            if product_name!='xrLST':
                filename = output_plot_dir+'{}_stack.tif'.format(product_name)
                with rasterio.open(filename, 'w', **profile) as dst:
                    dst.write(lst_stack)
                    print(filename)
                
            profile.update(count=1)
            
            filename = output_plot_dir+'{}_mean.tif'.format(product_name)
            with rasterio.open(filename, 'w', **profile) as dst:
                dst.write(lst_mean.reshape([1,lst_stack.shape[1],lst_stack.shape[2]]))
                print(filename)
                
            if product_name!='xrLST':
                filename = output_plot_dir+'{}_var_n.tif'.format(product_name)
                with rasterio.open(filename, 'w', **profile) as dst:
                    dst.write((lst_var/lst_count).reshape([1,lst_stack.shape[1],lst_stack.shape[2]]))
                    print(filename)
        

    print('All done')





    
if __name__ == '__main__':

    """ Default parameters   

    """

    date_label = '2014-2016'
    place_label = 'derbyshire'
    max_cloud = 70.0
    cloud_mask_bits = [0,1,4,8,10]

    if date_label=='2013-2014':
        dates_of_interest = [['20131101','20140228']]
    elif date_label=='2014-2016':
        dates_of_interest = [['20141101','20150228'],['20151101','20160228']]
    elif date_label=='2016-2018':
        dates_of_interest = [['20161101','20170228'],['20171101','20180228']]
    elif date_label=='2017-2018':
        dates_of_interest = [['20171101','20180228']]
    else:
        date_label=''

    if place_label == 'derbyshire':
        pathrows = [['202','023'],['203','023']]
    else:
        pathrows =[]
    
    params = (date_label,place_label,dates_of_interest,pathrows,max_cloud,cloud_mask_bits)

    main(*params)
