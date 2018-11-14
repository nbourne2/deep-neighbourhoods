"""
Calculate LST from Landsat images using NDVI and image-based atm correction

This is mostly a clone of land-surface-temperature.ipynb from the astrosat repo
The scene metadata are used to convert DNs to brightness temperature
The emissivity is calculated from Landsat-derived NDVI, and is used to convert
from surface radiance to LST.

See: https://github.com/astrosat/land-surface-temperature

NB edits: 
 - changed the way downsampling is done 
 - added args to the main function so that transform and crs can be checked
   since downsampling not required for Landsat if transform and crs are equal
 - pass metadata path as arg so that metadata keys can be used to convert 
   DNs to radiance 

"""
import re
from rasterio.warp import reproject, Resampling

TIRS_WAVELENGTH = 10.895e-6  # average wavelength of TIRS band
NDVI_SOIL = 0.2  # from Avdan et al.
NDVI_VEGETATION = 0.5  # from Avdan et al. 
EMISSIVITY_WATER = 0.991  # from Avdan et al. 
EMISSIVITY_SOIL = 0.991  # from Avdan et al. but subtracting surface roughness
EMISSIVITY_VEGETATION = 0.968  # from Avdan et al. but subtracting surface roughness
SURFACE_ROUGHNESS = 0.005  # from Avdan et al.

PLANK = 6.62607004e-34  # [J s]
BOLTZMANN = 1.38064852e-23  # [J/K]
LIGHT_SPEED = 299792458  # [m/s]

def read_band_raster(band_name, qa_raster=None):
    """Read band from Landsat product"""
    path = os.path.join(LANDSAT_SCENES_PATH, LANDSAT_PRODUCT_ID,
                        '{}_B{}.TIF'.format(LANDSAT_PRODUCT_ID, band_name))
    with rasterio.open(path) as src:
        raster = src.read(1)

    # Use QA band to mask out the image background
    if isinstance(qa_raster, np.ndarray):
        raster = np.ma.masked_where(qa == 1, raster)
        
    return raster


def int_or_float(s):
    """Convert string number to int or float"""
    try:
        return int(s)
    except ValueError:
        return float(s)
    
    
def read_metadata(path):
    """Read key/value pairs from Landsat metadata text file. Returns
    dictionary with abbreviated keys.
    
    NB change: pass metadata path as arg
    """
    
    with open(path) as f:
        metadata = f.read()
    
    key_lookup = {'RADIANCE_MAXIMUM_BAND_10': 'lmax',
                  'RADIANCE_MINIMUM_BAND_10': 'lmin',
                  'QUANTIZE_CAL_MAX_BAND_10': 'qcalmax',
                  'QUANTIZE_CAL_MIN_BAND_10': 'qcalmin',
                  'K1_CONSTANT_BAND_10': 'k1',
                  'K2_CONSTANT_BAND_10': 'k2'}
    
    output = dict()
    for key, short_key in key_lookup.items():
        value_string = re.search(key + ' = (.*)\n', metadata).group(1)
        value = int_or_float(value_string)
        output[short_key] = value
        
    return output


def write_raster(path, raster):
    """Write raster (numpy.ma.core.MaskedArray) as GeoTIFF"""
    # Use TIRS band to read profile from a source file
    src_path = os.path.join(LANDSAT_SCENES_PATH, LANDSAT_PRODUCT_ID,
                            '{}_B{}.TIF'.format(LANDSAT_PRODUCT_ID, 10))
    with rasterio.open(src_path) as src:
        profile = src.profile

    # Write raster with updated profile
    profile.update(dtype=raster.dtype.name,
                   compress='lzw')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write_band(1, raster.filled())
        

def calculate_radiance(tirs, metadata_path):
    """Calculate TOA spectral radiance from TIRS digital numbers"""
    m = read_metadata(metadata_path)
    return ((tirs - m['qcalmin']) *
            (m['lmax'] - m['lmin']) /
            (m['qcalmax'] - m['qcalmin']) +
            m['lmin'])


def calculate_brightness_temperature(radiance, metadata_path):
    """
    Calculate at-satellite brightness temperature (in Kelvin) 
    from TOA spectral radiance
    """
    m = read_metadata(metadata_path)
    return m['k2'] / np.log(m['k1'] / radiance + 1)


def calculate_ndvi(nir, red):
    """Calculate NDVI from near-infrared and red bands"""
    nir = nir.astype('float64')
    red = red.astype('float64')
    return (nir - red) / (nir + red)


def calculate_mixed_cover_emissivity(ndvi):
    """Calculate emissivity for mixture of soil and vegetation cover"""
    vegetation_proportion = (
        (ndvi - NDVI_SOIL) 
        / (NDVI_VEGETATION - NDVI_SOIL)
    )**2

    return (EMISSIVITY_VEGETATION * vegetation_proportion +
            EMISSIVITY_SOIL * (1 - vegetation_proportion) +
            SURFACE_ROUGHNESS)


def calculate_emissivity(ndvi):
    """Calculate emissivity from NDVI"""
    # TODO: Investigate other ways to derive emissivity from NDVI.
    x = ndvi.copy().filled()  # np.logical_and() requires np.ndarray
    conditions = [x < 0,
                  np.logical_and(x >= 0, x < NDVI_SOIL),
                  np.logical_and(x >= NDVI_SOIL, x <= NDVI_VEGETATION),
                  x > NDVI_VEGETATION]
    choices = [EMISSIVITY_WATER,
               EMISSIVITY_SOIL + SURFACE_ROUGHNESS,
               calculate_mixed_cover_emissivity(x),
               EMISSIVITY_VEGETATION + SURFACE_ROUGHNESS]
    emissivity = np.select(conditions, choices)  # apply discontinuous function
    return np.ma.array(emissivity, mask=ndvi.mask)  # reapply mask


def downsample_raster(raster, factor=3):
    """Reduce raster resolution by factor. Preserves array shape."""
    downsampled = cv2.resize(raster, None, fx=1/factor, fy=1/factor)
    upsampled = cv2.resize(downsampled, (raster.shape[1], raster.shape[0]))
    return np.ma.array(upsampled, mask=raster.mask)


def calculate_land_surface_temperature_NB(red, nir, tirs, 
                                          oli_transform, tirs_transform, 
                                          oli_crs, tirs_crs, 
                                          metadata_path, celsius=True):
    """Calculate land surface temperature from brightness temperature
    
    NB changes: pass in red/nir and TIRS transforms
    Then reproject the NDVI to the pixel grid of the TIRS
    """
    
    radiance = calculate_radiance(tirs, metadata_path)
    brightness_temperature = calculate_brightness_temperature(
        radiance, metadata_path)
    
    # Downsample NDVI from 30m to 90m to match TIRS (FIXME: This should be 100 m).
    # Otherwise the high resolution features in NDVI translate to 'features'
    # in the LST raster that have a higher resolution than TIRS.
    ndvi = calculate_ndvi(nir, red)
    #ndvi = downsample_raster(ndvi, factor=3)  
    
    emissivity = calculate_emissivity(ndvi)

    # NB instead of downsampling by a fixed factor, 
    # lets reproject to the TIRS pixel grid
    #if ((tirs_transform == oli_transform) and (tirs_crs == oli_crs)):
    if (tirs_crs == oli_crs):
        emi_regrid = emissivity
    else:
        print(tirs_transform)
        print(oli_transform)
        print(tirs_crs)
        print(oli_crs)
        print((tirs_transform == oli_transform,tirs_crs == oli_crs))
        reproject(
            emissivity, emi_regrid,
            src_transform = oli_transform,
            dst_transform = tirs_transform,
            src_crs = oli_crs,
            dst_crs = tirs_crs,
            resampling = Resampling.bilinear)
    
    
    rho = PLANK * LIGHT_SPEED / BOLTZMANN
    land_surface_temperature = (
                    brightness_temperature /
                    (1 + (TIRS_WAVELENGTH * brightness_temperature / rho) *
                     np.log(emi_regrid))
    ) 
    
    if celsius:
        return land_surface_temperature - 273.15
    else:
        return land_surface_temperature