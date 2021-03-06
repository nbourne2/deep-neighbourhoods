"""Global variables for builder.py"""

rootdir = '/Users/nathan.bourne/data/thermcert/'
tirband = '10'
qaband = 'QA'
bband = '2'
gband = '3'
rband = '4'
nband = '5'

products = ['xLST','rLST','rxLST','xrLST']
    
landcover_file = rootdir+'copernicus/land_cover/g100_clc12_V18_5.tif'
lsoa_file = rootdir+'uk_data/astrosat_data/lsoa_with_gas_and_electricity.geojson'
uk_counties_shapefile = rootdir+'uk_data/counties_and_ua_boundaries/'\
    +'Counties_and_Unitary_Authorities_December_2015_Generalised_Clipped_'\
    +'Boundaries_in_England_and_Wales.shp'

ceda_username = 'nbourne'