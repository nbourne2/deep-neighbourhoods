"""Some diagnostic plots for the LST builder"""

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
diagnostics = cf.diagnostics

def display_qamask(scene_url,output_plot_dir,cloud_mask_bits,qamask_sm_width,
                    qamask_sm_method,**aoi_kwargs):

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
    smw = qamask_sm_width
    smm = qamask_sm_method
    mask_occ_sm = ru.smooth_mask_qa(bqa_data,[1],smw,method=smm)
    mask_cloud_sm = ru.smooth_mask_qa(bqa_data,[0,4],smw,method=smm)
    mask_clcon_sm = ru.smooth_mask_qa(bqa_data,[6],smw,method=smm)
    mask_cicon_sm = ru.smooth_mask_qa(bqa_data,[12],smw,method=smm)
    mask_cscon_sm = ru.smooth_mask_qa(bqa_data,[8],smw,method=smm)
    mask_sncon_sm = ru.smooth_mask_qa(bqa_data,[10],smw,method=smm)

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
    mask_all = ru.smooth_mask_qa(bqa_data,cloud_mask_bits,qamask_sm_width,
                                     method=qamask_sm_method)

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
    plt.close(fig)
    return

def display_rgb(scene_url,output_plot_dir,cloud_mask_bits,qamask_sm_width,
                qamask_sm_method,plot_qamask=False,**aoi_kwargs):
    
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
    ax1=ax1.axes

    # Plot cloud mask outline
    if plot_qamask:
        # Combined mask of selected bits
        mask_all = ru.smooth_mask_qa(bqa_data,cloud_mask_bits,qamask_sm_width,
                                     method=qamask_sm_method)
        ax1.contour(mask_all[ymin:ymax,xmin:xmax],levels=[0.5],
                   colors='yellow',linewidths=0.5,antialiased=True)

    fig.savefig(filename)
    plt.close(fig)
    return

def display_agg_lst(lsoa_in_aoi,transform_aoi,aoi_bounds,lst_mean,landcover_mask1,landcover_mask2,output_file):

    lst_mean_masked_qa = lst_mean
    lst_mean_masked_lc1 = ma.array(lst_mean_masked_qa, mask=landcover_mask1).filled(fill_value=np.nan)
    lst_mean_masked_lc2 = ma.array(lst_mean_masked_qa, mask=landcover_mask2).filled(fill_value=np.nan)

    zs0 = zonal_stats(lsoa_in_aoi, lst_mean_masked_qa, affine=transform_aoi, stats=['mean','count'])
    zs1 = zonal_stats(lsoa_in_aoi, lst_mean_masked_lc1, affine=transform_aoi, stats=['mean','count'])
    zs2 = zonal_stats(lsoa_in_aoi, lst_mean_masked_lc2, affine=transform_aoi, stats=['mean','count'])

    labels = ['Aggregated LST Image',
              'Mean LST per LSOA masking non-residential urban land use',
              'Mean LST per LSOA masking all non-residential land use']

    cmap='afmhot'

    fig,axes = gpl.make_figure(shape=(1,3),figsize=(20,10))

    counter=-1
    for zs,im,lbl in zip([zs0,zs2,zs1],
                     [lst_mean_masked_qa,lst_mean_masked_lc2,lst_mean_masked_lc1],
                     labels):
        counter +=1
        zsmean = ma.array([z['mean'] for z in zs],
                          mask=[z['count']<=0 for z in zs],
                          fill_value=np.nan)
        
        # Add LST_mean column to gdf vector
        lsoa_new = lsoa_in_aoi.assign(LST_mean = zsmean.filled())
        
        # select the LSOA's that have some LST data: if don't do this,
        # the gpd.plot method doesn't deal with empty/nan/None values correctly.
        lsoa_new = lsoa_new.iloc[np.where(zsmean.mask==False)]
        
        # plot the masked LST map
        if counter==0:
            vmin,vmax = -6,6 #np.nanpercentile(im,[0,100])
            # print(vmin,vmax)
            base,nm = gpl.raster(axes[counter], im, 
                                 transform=None,
                                 colormap=cmap,
                                 vmin=vmin,
                                 vmax=vmax,
                                 title=lbl,
                                 titlesize=12,
                                 use_rst_plot=False,
                                 clean_data=False)    
        else:
            vmin,vmax = -6,6 #np.percentile(lsoa_new['LST_mean'].values,[0.01,100])
            base,nm = gpl.choropleth(axes[counter], lsoa_new,
                                     colname='LST_mean', 
                                     colormap=cmap,
                                     vmin=vmin,
                                     vmax=vmax,
                                     linewidth=0.2,
                                     title=lbl,
                                     titlesize=12,
                                     scheme='equal_interval')
            gpl.zoom_to_aoi(axes[counter], aoi_bounds)
            
        # Add colorbar to this plot
        cblbl = ('Relative LST / centigrade degrees' if counter==0 
                 else 'LSOA-averaged relative LST / centigrade degrees')
        CB0 = gpl.colorbar(axes[counter],nm,
                            colormap=cmap,
                            label=cblbl,
                            locbottom=False)

        
    fig.savefig(output_file)
    plt.close(fig)
    return

def display_lst_comparison(products,lst_data_dir,output_file):
    nprod = len(products)
    fig,axes = gpl.make_figure(shape=(1,nprod),figsize=(6.5*nprod,10))

    for ii in range(nprod):
        if nprod>1:
            ax=axes[ii]
        else:
            ax=axes
        filename = lst_data_dir+'{}_mean.tif'.format(products[ii])

        with rasterio.open(filename) as lstdata:
            lst_data1 = lstdata.read()[0,:,:]
            lst_trans1 = lstdata.transform
            
            ax,nm = gpl.raster(ax, lst_data1, transform=lst_trans1,
                       title=filename.split('/')[-1], colormap='afmhot')
            
            cb = gpl.colorbar(ax, nm, colormap='afmhot')

    plt.savefig(output_file)
    plt.close(fig)

    return
def display_lulc(landcover_data,landcover_mask1,landcover_mask2,output_plot_dir):
    # Original version of landcover map with full range of values from 0-50
    fig,axes = gpl.make_figure(shape=(1,3),figsize=(15,10))
    
    # im0 = rplt.show(landcover_data,ax=axes[0],vmin=1,vmax=50,title='land cover map')
    # im0 = axes[0].imshow(landcover_data,vmin=1,vmax=50,cmap='viridis')
    # axes[0].set_title('land cover map')
    # im1 = rplt.show(ma.array(landcover_data,mask=landcover_mask2),
    #           ax=axes[1],vmin=1,vmax=50,title='mask non-residential urban',cmap='viridis')
    # im2 = rplt.show(ma.array(landcover_data,mask=landcover_mask1),
    #           ax=axes[2],vmin=1,vmax=50,title='mask all non-residential',cmap='viridis')
    cmap = 'viridis'
    im0,nm0 = gpl.raster(axes[0], landcover_data, 
                           colormap=cmap,
                           vmin=1,
                           vmax=50,
                           title='land cover map',
                           use_rst_plot=True,
                           clean_data=False
                           )
    im1,nm1 = gpl.raster(axes[1], 
                           ma.array(landcover_data,
                                    mask=landcover_mask2), 
                           colormap=cmap,
                           vmin=1,
                           vmax=50,
                           title='mask non-residential urban',
                           use_rst_plot=True,
                           clean_data=False
                           )
    im2,nm2 = gpl.raster(axes[2], 
                           ma.array(landcover_data,
                                    mask=landcover_mask1), 
                           colormap=cmap,
                           vmin=1,
                           vmax=50,
                           title='mask all non-residential',
                           use_rst_plot=True,
                           clean_data=False
                           )
    

    # Add colorbar to this plot
    cbpos = [0.05,0.95,0.9,0.02]
    # inset0=fig.add_axes(cbpos,frameon=True,clip_on=True)
    # CB0 = fig.colorbar(im0, orientation='horizontal', cax=inset0)

    # adjust alignment of subplots...
    fig.subplots_adjust(left=0.05,right=0.95,
                        bottom=0.05,top=1,
                        wspace=0.05) 
    
    LC_classes = {'bounds':[0.5,2.5,11.5,21.5,25.5,37.5,44.5],
                  'midpts':[1.5,7,16.5,23.5,31.5,41],
                  'labels':['Urban Fabric',
                            'Industry/Commercial/etc',
                            'Agriculture',
                            'Forestry',
                            'Natural Vegetation',
                            'Water']
                  }
    tickvals = np.array(list(zip(LC_classes['bounds'],LC_classes['midpts']))).flatten()
    ticklabs = np.array(list(zip(['']*len(LC_classes['bounds']),LC_classes['labels']))).flatten()
    # CB0.set_ticks(tickvals)
    # CB0.set_ticklabels(ticklabs)
    
    CB0 = gpl.colorbar(im0,nm0,colormap=cmap,cbpos=cbpos)
    CB0.set_ticks(tickvals)
    CB0.set_ticklabels(ticklabs)
    
    fig.savefig(output_plot_dir+'Land_Cover_C30.pdf')
    plt.close(fig)
    # New version of landcover map with reduced number of discrete values given by classes above
    cmap = 'gist_earth_r'
    cmin=-2
    cmax=44
    fig,axes = gpl.make_figure(shape=(1,3),figsize=(15,10))
    
    landcover_data_disc = np.where(landcover_data < LC_classes['bounds'][0], 0,
                                   np.where(landcover_data < LC_classes['bounds'][1], LC_classes['midpts'][0],
                                           np.where(landcover_data < LC_classes['bounds'][2], LC_classes['midpts'][1],
                                                   np.where(landcover_data < LC_classes['bounds'][3], LC_classes['midpts'][2],
                                                           np.where(landcover_data < LC_classes['bounds'][4], LC_classes['midpts'][3],
                                                                   np.where(landcover_data < LC_classes['bounds'][5], LC_classes['midpts'][4],
                                                                           np.where(landcover_data < LC_classes['bounds'][6], LC_classes['midpts'][5], 50)
                                                                           )
                                                                   )
                                                           )
                                                   )
                                           )
                                  )

    # im0 = axes[0].imshow(landcover_data_disc,vmin=cmin,vmax=cmax,cmap=cmap)
    # axes[0].set_title('land cover map')
    # im1 = rplt.show(ma.array(landcover_data_disc,mask=landcover_mask2),
    #           ax=axes[1],vmin=cmin,vmax=cmax,title='mask non-residential urban',cmap=cmap)
    # im2 = rplt.show(ma.array(landcover_data_disc,mask=landcover_mask1),
    #           ax=axes[2],vmin=cmin,vmax=cmax,title='mask all non-residential',cmap=cmap)

    im0,nm0 = gpl.raster(axes[0], landcover_data_disc, 
                           colormap=cmap,
                           vmin=cmin,
                           vmax=cmax,
                           title='land cover map',
                           use_rst_plot=False,
                           )
    im1,nm1 = gpl.raster(axes[1], 
                           ma.array(landcover_data_disc,mask=landcover_mask2), 
                           colormap=cmap,
                           vmin=cmin,
                           vmax=cmax,
                           title='mask non-residential urban',
                           use_rst_plot=True,
                           )
    im2,nm2 = gpl.raster(axes[2], 
                           ma.array(landcover_data_disc,mask=landcover_mask1), 
                           colormap=cmap,
                           vmin=cmin,
                           vmax=cmax,
                           title='mask all non-residential',
                           use_rst_plot=True,
                           )

    # Add colorbar to this plot
    cbpos = [0.05,0.95,0.9,0.02]
    # inset0=fig.add_axes(cbpos,frameon=True,clip_on=True)
    #CB0 = fig.colorbar(im0, orientation='horizontal', cax=inset0)
    # CB0 = matplotlib.colorbar.ColorbarBase(inset0,orientation='horizontal',
    #                                       boundaries= LC_classes['bounds'],
    #                                       cmap=cmap,norm=im0.norm)
    
    CB0 = gpl.colorbar(im0,nm0,colormap=cmap,cbpos=cbpos,
                        discrete=True,
                        disc_bounds = LC_classes['bounds'],
                        disc_tickvals = tickvals,
                        disc_ticklabs = ticklabs
                      )
    
    # adjust alignment of subplots...
    fig.subplots_adjust(left=0.05,right=0.95,
                        bottom=0.05,top=1,
                        wspace=0.05) 

    # Add labels to the colorbar
    # CB0.set_ticks(tickvals)
    # CB0.set_ticklabels(ticklabs)
    
    # Do not show axis tick labels 
    # for ax in axes:
    #     ax.set_xticks([])
    #     ax.set_yticks([])

    fig.savefig(output_plot_dir+'Land_Cover_C6.pdf')
    plt.close(fig)
    #CB0 = fig.colorbar(im0, orientation='horizontal', ax=axes[0])
    #     cbpos = [0.28,0.92,0.25,0.02]
    #     inset2=fig.add_axes(cbpos,frameon=True,clip_on=True)
    #     im2 = ax1.imshow(lst_data_mask_all,cmap='CMRmap',vmin=-5,vmax=15)
    #     CB2 = fig.colorbar(im2, orientation='horizontal', ax=ax2, cax=inset2)
    #     CB2.set_label('Degrees Centigrade')

    #     cbpos = [0.4,0.92,0.2,0.02]
    #     cax=fig.add_axes(cbpos,frameon=True,clip_on=True)
    #     sm = plt.cm.ScalarMappable(cmap='Spectral', norm=plt.Normalize(vmin=vminf, vmax=vmaxf))
    #     sm._A = []
    #     fig.colorbar(sm, orientation='horizontal', cax=cax)

    return  

def display_urban_rural(lsoa_in_aoi,aoi_bounds,output_file):
    
    fig,ax = gpl.make_figure(figsize=(6,8))

    ax,nm = gpl.choropleth(
                ax,lsoa_in_aoi,colname='LC_urban',
                colormap='viridis',scheme='equal_interval',
                vmin=-0.8, vmax=1,
                edgecolor='k',linewidth=0.2,
                title='Urban/rural classification by dominant land cover'
                )
    gpl.zoom_to_aoi(ax,aoi_bounds)
    fig.savefig(output_file)
    plt.close(fig)

    return