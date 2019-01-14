"""
Simple script to combine the output LST columns from multiple geodataframes
"""

rootdir = ''

def main(lsoa_file_list,LST_column,output_filename):

    print('{} files to concatenate'.format(len(lsoa_file_list)))
    print('Column: {}'.format(LST_columns))

    all_lsoa_data = []
    for lsoa_file in lsoa_file_list:
        lsoa_data = gpd.read_file(lsoa_file)

        if len(all_lsoa_data)>0:
            assert all_lsoa_data[-1].crs == lsoa_data.crs

        all_lsoa_data += [lsoa_data[['geometry',LST_column]]]
        print('{} rows '.format(len(lsoa_data)))

    all_lsoa_data = pd.concat(all_lsoa_data,sort=False)
    print('Total {} rows'.format(len(all_lsoa_data)))

    all_lsoa_data.to_file(output_filename,driver='GeoJSON')
    print('Output to '+output_filename)

    return

if __name__ == '__main__':

    LST_column = 'rLST_mean'

    place_labels = ['derbyshire','nottinghamshire']
    date_label = '2015-2019'
    LST_label = 'rLST'
    lc_label = 'uaclc12'
    output_filename = rootdir+'output_LSOA_LST/Five_counties_lsoa_{}.geojson'.format(
        LST_column.split('_')[0])

    geojson_list = [
        rootdir+'output_LSOA_LST/{0}_{1}/lsoa_{0}_{2}_{3}.geojson'.format(
            place,date_label,LST_label,lc_label)
        for place in place_labels]

    main(geojson_list,LST_column,output_filename)