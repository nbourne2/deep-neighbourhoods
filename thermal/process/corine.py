"""Functions for dealing with CORINE land-use/land-cover maps"""

import rasterio
from rasterio.warp import calculate_default_transform, reproject, transform_bounds
from rasterio.transform import rowcol

from common import raster_utils as ru

def reproject_raster(landcover_file,scene_file,output_file):
    """
    Open Land Cover data and transform to TIR grid and clip
    """

    with rasterio.open(landcover_file) as landcover:
        with rasterio.open(scene_file) as scene:
            dst_crs = scene.crs
            src_crs = landcover.crs
            src_transform = ru.rst_transform(landcover)
            dst_transform = ru.rst_transform(scene)
    #         print(rst_transform(scene),scene.width,scene.height)
            
    #         transform, width, height = calculate_default_transform(
    #             src_crs, dst_crs, landcover.width, landcover.height, 
    #             *landcover.bounds)
    #         print(transform,width,height)
            
            scene_bounds_in_landcover = transform_bounds(
                dst_crs, src_crs, *scene.bounds)
            transform, width, height = calculate_default_transform(
                src_crs, dst_crs, landcover.width, landcover.height, 
                *scene_bounds_in_landcover,
                resolution=(dst_transform[0],abs(dst_transform[4]))
    #             dst_width= scene.width,
    #             dst_height= scene.height
            )
            # print(transform,width,height)
            # print(scene.width,scene.height)
            
            kwargs = landcover.meta.copy()
            kwargs.update({
                    'crs': dst_crs,
                    'transform': transform,
                    'width': width,
                    'height': height
            })

            with rasterio.open(output_file, 'w', **kwargs) as dst:
                for i in range(1, landcover.count + 1):
                    reproject(
                        source=rasterio.band(landcover, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src_transform,
                        src_crs=src_crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)
                    
            print(output_file)

            try:
                with rasterio.open(output_file) as src:
                    landcover_data = src.read()
                lst_data = scene.read()
                assert landcover_data.shape == lst_data.shape
            except:
                print(landcover_data.shape)
                print(lst_data.shape)
                import pdb; pdb.set_trace()
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

    return landcover_mask1,landcover_mask2

       
