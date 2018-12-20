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
params = ()
builder.main(params)


Dependencies:
land_surface_temperature
raster_utils
landsat



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
from common import raster_utils as ru
from common import geoplot as gpl

# GLOBALS
rootdir = '/Users/nathan.bourne/data/thermcert/'
tirband = '10'
qaband = 'QA'
bband = '2'
gband = '3'
rband = '4'
nband = '5'

landcover_file = rootdir+'copernicus/land_cover/g100_clc12_V18_5.tif'
lsoa_file = rootdir+'uk_data/astrosat_data/lsoa_with_gas_and_electricity.geojson'
uk_counties_shapefile = rootdir+'uk_data/counties_and_ua_boundaries/'\
    +'Counties_and_Unitary_Authorities_December_2015_Generalised_Clipped_'\
    +'Boundaries_in_England_and_Wales.shp'

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


def display_qamask(scene_url,output_plot_dir,**aoi_kwargs):

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
        ru.mask_qa(bqa_data,bits=[0,1,4,8,10]),
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
    content = f.readline()
    f.close()
    return content

def stack_tir(scene_urls,aoi,aoi_crs,
              subtract_median_lst=True,subtract_air_temp=False):
    """
    Convert clipped and masked TIR scenes to LST, then aggregate 
    Also allow for calculation of xLST: subtracting mean air temp 
    on day of acquisition, and aggregating the result

    This has to be re-run each time you want to produce xLST,rxLST,rLST stacks

    """
    if subtract_air_temp:
        from script import met_climate
        import imp
        imp.reload(met_climate)

        ceda_password = get_ceda_password()
        ceda_username = 'nbourne'

        at = met_climate.access_ukcp09(ceda_username,ceda_password)

    
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

        with rasterio.open(scene_bqa) as bqa:
            bqa_data,bqa_trans = ru.read_in_aoi(bqa,**aoi_kwargs)

            with rasterio.open(scene_tir) as tir:
                tir_data,tir_trans = ru.read_in_aoi(tir,**aoi_kwargs)
                tir_crs = tir.crs

            with rasterio.open(scene_red) as red:
                red_data,red_trans = ru.read_in_aoi(red,**aoi_kwargs)
                red_crs = red.crs

            with rasterio.open(scene_nir) as nir:
                nir_data,nir_trans = ru.read_in_aoi(nir,**aoi_kwargs)

            bqa_data = ma.array(bqa_data[0,:,:].squeeze())
            tir_data = ma.array(tir_data[0,:,:].squeeze())
            red_data = ma.array(red_data[0,:,:].squeeze())
            nir_data = ma.array(nir_data[0,:,:].squeeze())
            
            print(np.shape(tir_data))


            
            # Determine size of stack allowing for AoI to extend outside of scene
            if counter == 0:
                aoi_box = rasterio.warp.transform_bounds(aoi_crs,tir_crs,*aoi.values())
                aoi_box = dict(zip(('minx','miny','maxx','maxy'),aoi_box))
                aoi_left, aoi_bottom, aoi_right, aoi_top = aoi_box
                rowmin,colmin = (bqa.index(aoi_left,aoi_top,op=round))
                rowmax,colmax = (bqa.index(aoi_right,aoi_bottom,op=round))
                stack_height,stack_width = (rowmax-rowmin,colmax-colmin)
                lst_stack = (ma.zeros((len(scenes_bqa),stack_height,stack_width),
                                      dtype=np.float,fill_value=np.nan
                                     )+np.nan)   

            # Determine size of intersect in THIS scene
            intersect = ru.aoi_scene_intersection(aoi_box,bqa)
            ins_left, ins_bottom, ins_right, ins_top = intersect.bounds
            rowmin,colmin = (bqa.index(ins_left,ins_top,op=round))
            rowmax,colmax = (bqa.index(ins_right,ins_bottom,op=round))
            # Is this still correct given that we read a window?

        # Read data 
        bqa_data = ma.array(bqa_data[0,rowmin:rowmax,colmin:colmax])
        tir_data = ma.array(tir_data[0,rowmin:rowmax,colmin:colmax])
        red_data = ma.array(red_data[0,rowmin:rowmax,colmin:colmax])
        nir_data = ma.array(nir_data[0,rowmin:rowmax,colmin:colmax])

        lst_data = lst.calculate_land_surface_temperature_NB(
                        red_data, nir_data, tir_data,
                        red_trans, tir_trans, 
                        red_crs, tir_crs, scene_metadata
                        )
        # Have to pass it a path to a local tmp file "scene_metadata"


        # Masks
        smw = 11
        mask_all = filters.maximum_filter(
                            ru.mask_qa(bqa_data,bits=[0,1,4,8,10]),size=smw
                            )

        lst_data_mask_all = ma.array(lst_data,
                            mask=mask_all,
                            dtype=np.float,
                            fill_value=np.nan) #.filled()

        # After masking, reproject
        # Actually I don't think this is necessary if they share a CRS
        if counter > 0:
            assert tir.crs == prev_crs
        prev_crs = tir.crs

        # Now do some normalisation
        if subtract_air_temp:
            filename = scene_tir.split('/')[-1]
            datestring = filename.split('_')[3]
            # month = int(datestring[4:6])
            atdata = at.grid_temp_over_scene(
                tir, datestring, interpolation='linear')
            atdata = atdata[rowmin:rowmax,colmin:colmax]
            assert lst_data_mask_all.shape == atdata.shape
#                         lst_data_mask_all = lst_data_mask_all - atdata
            lst_data_mask_all = ma.array(
                            lst_data_mask_all - atdata,
                            mask=mask_all,
                            fill_value=np.nan)

            colorbar_label = 'Degrees relative to air temp on this day'
            colorbar_vmin,colorbar_vmax = -5,15
            
            if subtract_median_lst:
                # ALSO subtract median xLST
                medval = ma.median(lst_data_mask_all)
#                             lst_data_mask_all = lst_data_mask_all - medval
                lst_data_mask_all = ma.array(
                            lst_data_mask_all - medval,
                            mask=mask_all,
                            fill_value=np.nan)
                
                colorbar_label = 'Degrees relative to air temp on this day, subtracting median'
                colorbar_vmin,colorbar_vmax = -5,15
            
#                         fig,ax1 = plt.subplots(figsize=(10,10))
#                         cbpos = [0.28,0.92,0.25,0.02]
#                         inset2=fig.add_axes(cbpos,frameon=True,clip_on=True)

#                         im2 = ax1.imshow(atdata,cmap='CMRmap')
#                         CB2 = fig.colorbar(im2, orientation='horizontal', ax=ax1, cax=inset2)
#                         CB2.set_label('Deg C')

#                         fig.suptitle('Air Temp',fontsize='large')

        elif subtract_median_lst:
            # Subtract median LST from scene (within QA mask) 
        
            medval = ma.median(lst_data_mask_all)
#                         lst_data_mask_all = lst_data_mask_all - medval
            lst_data_mask_all = ma.array(
                        lst_data_mask_all - medval,
                        mask=mask_all,
                        fill_value=np.nan)

            colorbar_label = 'Degrees relative to air temp on this day, subtracting median'
            colorbar_vmin,colorbar_vmax = -5,15
    
            colorbar_label = 'Degrees relative to median'
            colorbar_vmin,colorbar_vmax = -5,15
        
        else:
            colorbar_label = 'Degrees Centigrade'
            colorbar_vmin,colorbar_vmax = -5,15
        
        # Then add to stack
        lst_stack[counter,:,:] = lst_data_mask_all

    return lst_stack

def main(*args,diagnostics=False):
    """Main function
    
    Input:
    par = (                 # E.g.
        place_label,        # 'derbyshire'
        date_label,         # '2014-2016'
        dates_of_interest,  # [['20141101','20150228'],['20151101','20160228']]
        pathrows            # [['202','023'],['203','023']]
        )
    diagnostics = whether to make images showing the QA masks and RGB images 
        of each scene

    """
    date_label,place_label,dates_of_interest,pathrows,max_cloud = args
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
    counties = gpd.read_file(uk_counties_shapefile) 
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

    rlst_stack = stack_tir(scene_urls,county_bounds,county_crs,
                           subtract_median_lst=True,subtract_air_temp=False
                           )
                           








    
if __name__ == '__main__':

    """ Default parameters   

    """

    date_label = '2014-2016'
    place_label = 'derbyshire'
    max_cloud = 70.0

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
    
    params = (date_label,place_label,dates_of_interest,pathrows,max_cloud)

    main(*params)
