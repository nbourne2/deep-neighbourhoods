"""Read Energy Consumption data and compute energy efficiency estimates

Inputs: 
Annual energy consumption by LSOA geojson
LST data geojson

Return: geojoson containing all energy consumption data and 
energy efficiency estimates

"""

# Imports 
import numpy as np
import pandas as pd
import geopandas as gpd

# Globals
rootdir = '/Users/nathan.bourne/data/thermcert/'
consumption_file = rootdir+'uk_data/astrosat_data/lsoa_with_gas_and_electricity.geojson'
    
def get_estimates(lst_file):
    """Main function
    
    Input:
    lst_file:   Either a path to a geojson file 
                OR a geodataframe containing LST_mean column

    Return:
    geodataframe containing LST, DTR and energy consumption data
    """

    lsoa_all = read_data(lst_file)

    lsoa_all = define_params(lsoa_all)

    return lsoa_all

def read_data(lst_file):
    """Read data on energy consumption and LST for a given place, year, product
    
    Join on LSOA code, not spatial join.

    Return:
    Merged geodataframe (with geometry from the LST, not the energy cons.)
    """

    lsoa_consumption = gpd.read_file(consumption_file)

    if type(lst_file) is str:
        lsoa_lst = gpd.read_file(lst_file) 
    else:
        lsoa_lst = lst_file

    lsoa_all = lsoa_lst.merge(
                         lsoa_consumption.drop('geometry',axis=1),
                         # lsoa_consumption[['lsoa_code',
                         #                   'accommodation',
                         #                   'gas_consumption',
                         #                   'gas_meters',
                         #                   'electricity_consumption',
                         #                   'electricity_meters'
                         #                   ]],
                         left_on='LSOA code',
                         right_on='lsoa_code')

    return lsoa_all

def define_params(lsoa_all):
    """Define energy efficiency parameters and add to dataframe

    Input: dataframe with the following columns:
    accommodation = number of homes
    gas_meters = number of gas meters
    gas_consumption 
    electricity_consumption
    central_heating_gas_pct
    central_heating_electric_pct
    LST_mean

    Return: same dataframe with additional energy efficiency (DTR) columns
    """

    
    Frac_on_gas_grid = (pd.to_numeric(lsoa_all['gas_meters'].values
                                      ,errors='coerce') 
                        / pd.to_numeric(lsoa_all['accommodation'].values
                                        ,errors='coerce') 
    )
    Frac_on_gas_grid[Frac_on_gas_grid>1] = 1
                                      
    LST = pd.to_numeric(lsoa_all['LST_mean'],errors='coerce')
    GC = pd.to_numeric(lsoa_all['gas_consumption'],errors='coerce') 
    EC = pd.to_numeric(lsoa_all['electricity_consumption'],errors='coerce') 

    # These must be normalised otherwise they can't be added together
    GC = GC / (np.max(GC))
    EC = EC / (np.max(EC))
    GCC = GC / Frac_on_gas_grid

    DTR1 = (10+LST) / GC
    DTR2 = (10+LST) / GCC

    Frac_main_heat_gas = (pd.to_numeric(lsoa_all['central_heating_gas_pct']
                                        ,errors='coerce') 
                            /100.0)
    Frac_main_heat_elec = (pd.to_numeric(lsoa_all['central_heating_electric_pct']
                                         ,errors='coerce') 
                            /100.0)
    Frac_main_heat_other = 1.0 - Frac_main_heat_gas - Frac_main_heat_elec
                        
    assert max(Frac_main_heat_other) <= 1
    assert min(Frac_main_heat_other) >= 0

    TC = ((GC * Frac_main_heat_gas) + (EC * Frac_main_heat_elec)) / (1 - Frac_main_heat_other)
                        
    DTR3 = (10+LST) / TC

    lsoa_all = lsoa_all.copy().assign(DTR1 = DTR1,
                                      DTR2 = DTR2,
                                      DTR3 = DTR3)

    return lsoa_all



