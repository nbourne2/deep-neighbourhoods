import numpy as np

def NDWI(green, nir):
    """Calculates water index

    parameters
    ----------
    green: GREEN band as input
    red: NIR band as input
    """
    NDWI = (green.astype('float') - nir.astype('float')) / (green.astype('float') + nir.astype('float'))
    NDWI = np.where(NDWI>=0.18, np.nan, 1)
    return NDWI


