"""
This script performs texture analysis using snappy. Snappy is a python module that allows users to access SNAP java API
from python. 
"""

import sys
sys.path.append(
    '/home/io/ASTROSAT/code/urban_extraction/classification_module')

import os
from constants import DATA_PATH
# SNAP's Graph Processing Framework GPF used for developing and executing raster data operators and graphs of such operators.
from snappy import GPF
# The ProductIO class provides several utility methods concerning data I/O for remote sensing data products.
from snappy import ProductIO
from snappy import HashMap
from snappy import jpy

# JAVA - python bridge
HashMap = jpy.get_type('java.util.HashMap')
parameters = HashMap()

def read(filename):
    return ProductIO.readProduct(filename)

def write(product, filename):
    ProductIO.writeProduct(product, filename, "BEAM-DIMAP")


def resampling(product):
    """Returns a resampled product with same pixel size for all bands. 
    parameters
    ----------
    product: Source product
    """
    parameters.put("referenceBand", "B2")
    parameters.put("upsampling", "Nearest")
    parameters.put("downsampling", "First")
    parameters.put("flagDownsampling", "First")
    parameters.put("resampleOnPyramidLevels", "true")
    result = GPF.createProduct("Resample", parameters, product)
    return result


def glcm(product, param):
    """Returns the five GLCM texture measures 

    parameters
    ----------
    product: The result derived from the resampling process
    param: Bands to be used for texture analysis
    """
    parameters.put("windowSize", "5x5")
    parameters.put("displacement", "1")
    parameters.put("quantizationLevels", "32")
    parameters.put("outputCorrelation", "true")
    parameters.put("outputHomogeneity", "true")
    parameters.put("outputDissimilarity", "true")
    parameters.put("outputMean", "true")
    parameters.put("outputVariance", "true")
    parameters.put("outputASM", "false")
    parameters.put("outputContrast", "false")
    parameters.put("outputEnergy", "false")
    parameters.put("outputEntropy", "false")
    parameters.put("outputMAX", "false")
    for para_name in para_dict:
        parameters.put(para_name, para_dict[para_name])
        result = GPF.createProduct("GLCM", parameters, product)
    return result

def glcm_computation(file_name):
    """Writes the GLCM product 

    parameters
    ----------
    file_name: Sentinel-2 raw data
    """
    product = read(inputpath)
    resample = resampling(product)
    GLCM = glcm(resample, para_dict)
    write(GLCM, os.path.join(DATA_PATH,'output','S2_GLCM.dim'))


if __name__ == '__main__':
    inputpath = os.path.join(
        DATA_PATH, 'input', 'S2A_MSIL2A_20190101T114501_N0211_R123_T30UVG_20190101T121651.SAFE/MTD_MSIL2A.xml')
    # bands to be used for texture analysis
    para_dict = {'sourceBands': 'B3,B5,B6'}
    glcm_computation(inputpath)





