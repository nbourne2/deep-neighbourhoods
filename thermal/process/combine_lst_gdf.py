"""
Simple script to combine the output LST columns from multiple geodataframes
"""
import geopandas as gpd
import pandas as pd
import glob
import os
import numpy as np
import config as cf
from shapely.geometry import box

rootdir = cf.rootdir#'/Users/nathan.bourne/data/thermcert/'
new_colname = 'residential_heat_loss'
uk_counties_shapefile = cf.uk_counties_shapefile
#rootdir+'uk_data/counties_and_ua_boundaries/'\
#    +'Counties_and_Unitary_Authorities_December_2015_Generalised_Clipped_'\
#    +'Boundaries_in_England_and_Wales.shp'

def main(lsoa_file_list,place_labels,LST_column,output_filename):

    print('{} files to concatenate'.format(len(lsoa_file_list)))
    print('Column: {}'.format(LST_column))

    counties = gpd.read_file(uk_counties_shapefile)

    all_lsoa_data = []
    for ii in range(len(lsoa_file_list)):
        lsoa_data = gpd.read_file(lsoa_file_list[ii])

        if len(all_lsoa_data)>0:
            assert all_lsoa_data[-1].crs == lsoa_data.crs

        # Select LSOAs in County
        counties = counties.to_crs(lsoa_data.crs)
        county_ind = np.where(counties['ctyua15nm'].str.match(
            place_labels[ii]+'*',case=False,na=False))[:][0]
        
        # Check for other features contained within this county
        # print(county_ind[0],counties['ctyua15nm'].iloc[county_ind[0]])
        county_bounds = box(*counties['geometry'].iloc[county_ind[0]].bounds)
        all_ind_in_county = np.where([
                        county_bounds.contains(g)
                        for g in counties['geometry'].values])[:][0]
        # print(other_ind)
        # for o in other_ind:
        #     print(o,counties.iloc[o]['ctyua15nm'])

        lsoa_data = gpd.sjoin(lsoa_data,
                              counties.iloc[all_ind_in_county],
                              how='inner')

        all_lsoa_data += [lsoa_data[['geometry',LST_column]]]
        print('{} rows in {}'.format(len(lsoa_data),place_labels[ii]))

    all_lsoa_data = pd.concat(all_lsoa_data,sort=False)
    print('Total {} rows'.format(len(all_lsoa_data)))

    # all_lsoa_data = all_lsoa_data.rename(columns={LST_column:new_colname})
    all_lsoa_data[new_colname] = pd.to_numeric(all_lsoa_data[LST_column],
                                               errors='coerce')
    all_lsoa_data = all_lsoa_data.drop(columns=LST_column)
    print('Column renamed to {}'.format(new_colname))

    if len(glob.glob(output_filename)):
        print('Overwriting output file')
        os.system('rm {}'.format(output_filename))
    all_lsoa_data.to_file(output_filename,driver='GeoJSON')
    print('Output to '+output_filename)

    return

if __name__ == '__main__':

    LST_column = 'rLST_mean'

    place_labels = ['derbyshire','nottinghamshire','shropshire','staffordshire','herefordshire']
    date_label = '2015-2019'
    LST_label = 'rLST'
    lc_label = 'uaclc12'
    output_filename = rootdir+'output_LSOA_LST/Five_counties_lsoa_{}.geojson'.format(
        LST_column.split('_')[0])

    geojson_list = [
        rootdir+'output_LSOA_LST/{0}_{1}/lsoa_{0}_{2}_{3}.geojson'.format(
            place,date_label,LST_label,lc_label)
        for place in place_labels]

    main(geojson_list,place_labels,LST_column,output_filename)