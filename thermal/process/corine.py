"""Functions for dealing with CORINE land-use/land-cover maps"""

import os 
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
from rasterio.transform import rowcol

from common import raster_utils as ru

def reproject_raster(landcover_file,scene_file,output_file):
    """
    Open Land Cover data and transform to TIR grid and clip

    This is a bit more convoluted than I would like
    But the problem is that rasterio.warp.reproject and/or 
    calculate_default_transform do not extract exactly the AoI you want
    but seem to include some buffer around the edge.

    The only way I can find to end up with a Landcover map with the 
    same resolution, transform and shape as the original scene is to
    first reproject and write to an intermediate file, then open
    a window on that file and rewrite to the final file.
    """

    with rasterio.open(landcover_file) as landcover:
        with rasterio.open(scene_file) as scene:

            # First reproject the landcover map to the scene's crs & transform
            dst_crs = scene.crs
            src_crs = landcover.crs
            src_transform = ru.rst_transform(landcover)
            dst_transform = ru.rst_transform(scene)
         
            scene_bounds_in_landcover = transform_bounds(
                dst_crs, src_crs, *scene.bounds)

            
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, landcover.width, landcover.height, 
                *scene_bounds_in_landcover,
                resolution=(dst_transform[0],abs(dst_transform[4]))
            )
            # print(transform,width,height)
            # print(scene.width,scene.height)

            scene_aoi = dict(zip(('minx','miny','maxx','maxy'),scene.bounds))
            scene_shape = (scene.width,scene.height)

        # Then output the result to an intermediate file.
        kwargs = landcover.meta.copy()
        kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height
        })
        intermed_file = output_file.replace('.tif','_temp.tif')
        with rasterio.open(intermed_file, 'w', **kwargs) as intermed:
            for i in range(1, landcover.count + 1):
                reproject(
                    source=rasterio.band(landcover, i),
                    destination=rasterio.band(intermed, i),
                    src_transform=src_transform,
                    src_crs=src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
            
            intermed_shape = (intermed.width,intermed.height)
            
    # Now check if the size is correct  
    if (intermed_shape == scene_shape):
        os.system('mv {} {}'.format(intermed_file,output_file))
    else:
        # print((intermed.width,intermed.height))
        # print((scene.width,scene.height))
        # print('rewriting windowed file')

        # Read intermediate file using window on scene
        with rasterio.open(intermed_file, 'r') as intermed:
            landcover_data,landcover_trans = ru.read_in_aoi(
                intermed,aoi=scene_aoi,aoi_crs=dst_crs)
            # print(landcover_data.shape)
            profile = intermed.profile
            profile.update(
                width=landcover_data.shape[2],
                height=landcover_data.shape[1],
                transform=landcover_trans,
            )
            # print(profile)

        with rasterio.Env():
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(landcover_data)

        os.system('rm {}'.format(intermed_file))
    return 


def mask_raster(lulc_clip_file, output_plot_dir):
    """
    Define Land Use mask from clipped Land Cover raster

    2 masks:
    mask1 = only keep urban residential classes (1,2)
    mask2 = discard urban non-residential (3-11)
    """
    
    mask1_lc_minvalid = 1
    mask1_lc_maxvalid = 2
    mask2_lc_minvalid1 = 1
    mask2_lc_maxvalid1 = 2
    mask2_lc_minvalid2 = 12
    mask2_lc_maxvalid2 = 50
    
    with rasterio.open(lulc_clip_file) as landcover:
        landcover_data = landcover.read()
        landcover_data = landcover_data.squeeze()
        # print(landcover_data.shape)
        # print(landcover.transform)
        # print(landcover.crs)
        # print(landcover.bounds)
        
        # rowmin,colmin = (landcover.index(aoi_left,aoi_top,op=round))
        # rowmax,colmax = (landcover.index(aoi_right,aoi_bottom,op=round))
        # print(rowmin,colmin)
        # print(rowmax,colmax)

        # landcover_data = landcover_data[0,rowmin:rowmax,colmin:colmax]
        # Not necessary if the assert statement in the previous function passed
        

    landcover_mask1 = np.array(
        ~((landcover_data>=mask1_lc_minvalid) 
            & (landcover_data<=mask1_lc_maxvalid)))
    landcover_mask2 = np.array(
        ~(  ((landcover_data>=mask2_lc_minvalid1) 
                & (landcover_data<=mask2_lc_maxvalid1))
            | (landcover_data>=mask2_lc_minvalid2) 
            & (landcover_data<=mask2_lc_maxvalid2)
         )
    )

    return landcover_data,landcover_mask1,landcover_mask2

       
