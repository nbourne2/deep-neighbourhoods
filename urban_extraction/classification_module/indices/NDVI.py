import numpy as np

def NDVI(nir,red):
    """Calculates NDVI index

    parameters
    ----------
    nir: NIR band as input
    red: RED band as input
    """
    NDVI = (nir.astype('float') - red.astype('float')) / (nir.astype('float') + red.astype('float'))
    NDVI = np.where(NDVI>=0.5, np.nan, 1)
    return NDVI





