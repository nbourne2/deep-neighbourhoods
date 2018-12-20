"""Utilities for raster manipulation"""

import numpy as np
import rasterio
from affine import Affine
from skimage.exposure import equalize_adapthist
from shapely.geometry import box


def read_in_aoi(src,aoi=None,aoi_crs=None):
    """
    Read an open rasterio dataset object within a 
    window defined by a given AoI in any crs
    
    Return the windowed data array and window transform
    If no aoi or no aoi_crs, return full array and transform
    """
    if (aoi is None) or (aoi_crs is None):
       arr = src.read()
       wtran = rst_transform(src)
    else:
        aoi_bounds = rasterio.warp.transform_bounds(aoi_crs,src.crs,*aoi.values())
        aoi_bounds=dict(zip(('minx','miny','maxx','maxy'),aoi_bounds))
        AoIrow0,AoIcol0 = rasterio.transform.rowcol(
            src.transform,aoi_bounds['minx'],aoi_bounds['maxy'])
        AoIrow1,AoIcol1 = rasterio.transform.rowcol(
            src.transform,aoi_bounds['maxx'],aoi_bounds['miny'])
        AoIheight,AoIwidth = AoIrow1 - AoIrow0, AoIcol1 - AoIcol0
        window = rasterio.windows.Window(AoIcol0,AoIrow0,AoIwidth,AoIheight)
        
        arr = src.read(window=window)
        wtran = src.window_transform(window)
    return arr,wtran


def rst_transform(dst):
    """
    Extract correct transform data from rasterio dataset 
    
    Compatible with both rasterio versions 0.36 and 1.0
    """
    if isinstance(dst.transform, Affine):
        #print('transform')
        return dst.transform
    else:
        #print('affine')
        return dst.affine


def aoi_scene_intersection(aoi, scene):
    """
    Compute intersection between a given AOI and Scene

    Inputs:
    aoi = geopandas Series given by df.bounds.loc[i]
    scene = rasterio dataset

    Returns:
    shapely Polygon
    """
    scnbox = box(minx=scene.bounds.left,maxx=scene.bounds.right,miny=scene.bounds.top,maxy=scene.bounds.bottom)
    aoibox = box(minx=aoi.minx,maxx=aoi.maxx,miny=aoi.miny,maxy=aoi.maxy)
    intersect = scnbox.intersection(aoibox)
    return intersect


def mask_qa(ints, bitmask=0b10011,bits=[]):
    """
    Define the mask for the QA pixels
    
    Inputs:
    bitmask should be 1 for each bit that you want to mask
    the return array will be True for values you want to mask      
    """

    if len(bits)>0:
        bitmask=np.sum([2**x for x in bits])
    return ((np.ravel(ints) & bitmask)>0).reshape(np.shape(ints))


"""functions to rescale data in images e.g. linear or hist equalise"""

def clahe_equalize(image, kernel_size=200, clip_limit=0.03):
    """Perform local adaptive histogram equalisation of an image"""

    if len(image.shape) == 3:
        # assume the smallest dimension is the band axis
        # roll this to the first axis, then put it back afterwards
        band_axis = np.argmin(image.shape) 
        if band_axis>0:
            image_in = np.rollaxis(image,band_axis)
        else:
            image_in = image
        image_out = np.zeros_like(image_in).astype(np.float64)
        for band in range(image.shape[band_axis]):
            image_out[band] = equalize_adapthist(
                image_in[band], kernel_size=kernel_size, clip_limit=clip_limit
            )
        if band_axis>0:
            image_out = np.moveaxis(image_out,0,band_axis)
    else:
        image_out = equalize_adapthist(
                image, kernel_size=kernel_size, clip_limit=clip_limit
        )
    return image_out     

def rescale_image(arr,kind='linear',**kwargs):
    """Rescale array values using various alternative methods"""
    if kind=='linear':
        return (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))

    if kind=='log':
        larr = np.log(arr)
        return (larr-np.nanmin(larr))/(np.nanmax(larr)-np.nanmin(larr))
    
    if kind=='sqrt':
        larr = np.sqrt(arr)
        return (larr-np.nanmin(larr))/(np.nanmax(larr)-np.nanmin(larr))
    
    if kind=='power2':
        larr = arr**2
        return (larr-np.nanmin(larr))/(np.nanmax(larr)-np.nanmin(larr))
                                       
    if kind=='hist_eq':
        pixels = arr.flatten()
        #pixels = (pixels-np.nanmin(pixels))/(np.nanmax(pixels)-np.nanmin(pixels))*256
        #cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), density=True, cumulative=True)
        cdf, bins = np.histogram(pixels, bins=256, 
                                 range=(np.nanmin(pixels),np.nanmax(pixels)),
                                 density=True )
        #print(cdf)
        cdf = np.cumsum(cdf)
        #print(cdf)
        #print(bins)
        new_pixels = np.interp(pixels, bins[:-1], cdf*255)
        return new_pixels.reshape(arr.shape)
        
    if kind=='clahe':
        # if arr.dtype == float:
        larr = arr.astype(np.float)
        larr = (larr-np.nanmin(larr))/(np.nanmax(larr)-np.nanmin(larr))
        return clahe_equalize(larr, **kwargs)


