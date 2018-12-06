""" Validate LST results against EPCs

Based on:
LSOA_EPC_aggregator.ipynb (Aggregation)
EPC_LST_Comparison.ipynb (Validation)

Procedure:
- Have to define date ranges (i) for landsat images and (ii) for filtering 
    EPCs (aggregate landsat raster should already exist)
- Have to define any additional filtering of EPCs, for example on transaction
    type
- Efficiently read EPC files (eg only read each file once)
- Match EPCs with postcode geodata
- Match postcode with LSOA
- Filter and aggregate EPC data per LSOA
- Output plots and correlation results in table

Calling:
    command line: 
        python validate_epc.py
    python: 
        import validate_epc; validate_epc.main()
    custom: 
        import validate_epc
        validate_epc.aggregate_epcs(...)
        run_validation(lsoa_with_epc,output_dir)

Possible values of EPC Transaction Type:
    assessment for green deal       Occurs before energy efficiency measures
    ECO assessment                  Occurs before energy efficiency measures
    FiT application                 Occurs before renewable energy measures
    following green deal            Occurs after energy efficiency measures
    marketed sale                   
    new dwelling
    NO DATA!
    non marketed sale
    none of the above
    not recorded
    rental
    rental (private)
    rental (private) - this is for backwards compatibility...
    rental (social)
    RHI application                 Occurs before renewable energy measures
    unknown

    see: https://www.ofgem.gov.uk/environmental-programmes
"""

# Imports
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import time
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress

from .common import geoplot
from . import estimate_energy_eff as eee

# Globals
rootdir = '/Users/nathan.bourne/data/thermcert/'
raw_data_dir = rootdir+'uk_data/astrosat_data/uk-stats/data/raw/'
pc_cent_file = rootdir+'uk_data/postcode_centroids/ONS_Postcode_Directory_Latest_Centroids.csv'
lad_file = raw_data_dir+'geometries/lad/Local_Authority_Districts_December_2016_Generalised_Clipped_Boundaries_in_the_UK.shp'

use_medsub_lst = True 
product_label = ('rLST' if use_medsub_lst else 'LST')

# Which columns of the EPC tables we are interested in
keepcols_epc = ['POSTCODE',
                'INSPECTION_DATE',
                'CURRENT_ENERGY_EFFICIENCY',
                'ROOF_ENERGY_EFF',
                'WALLS_ENERGY_EFF',
                'WINDOWS_ENERGY_EFF']
    
# ============================================================================
""" Main caller function"""

def main():
    """ Run the validation for a predefined set of validation samples"""

    green_deal_epcs = ['assessment for green deal',
                       'ECO assessment',
                       'following green deal',
                       ]

    new_build_epcs = 'new dwelling'
    
    experiments = [
        # 1a. 24 months of EPCs, all
        {
            'place' : 'derbyshire',
            'season' : '2013-2014' ,
            'epc_from' : '2013-01-01',
            'epc_to' : '2014-12-31',
            'epc_filter_trans' : '' ,
            'validation_label' : '24month_all'
        },
        # 1b. 24 months of EPCs, no new builds
        {
            'place' : 'derbyshire',
            'season' : '2013-2014' ,
            'epc_from' : '2013-01-01',
            'epc_to' : '2014-12-31',
            'epc_filter_trans' : new_build_epcs,
            'validation_label' : '24month_existing'
        },
        # 2a. 12 months of EPCs, all
        { 
            'place' : 'derbyshire',
            'season' : '2013-2014' ,
            'epc_from' : '2013-04-01',
            'epc_to' : '2014-03-31',
            'epc_filter_trans' : '' ,
            'validation_label' : 'apr-mar_all'
        },
        # 2b. 12 months of EPCs, no green-deal etc
        {
            'place' : 'derbyshire',
            'season' : '2013-2014' ,
            'epc_from' : '2013-04-01',
            'epc_to' : '2014-03-31',
            'epc_filter_trans' : green_deal_epcs,
            'validation_label' : 'apr-mar_noECO'
        },
        # 2c. 12 months of EPCs, no new builds
        {
            'place' : 'derbyshire',
            'season' : '2013-2014' ,
            'epc_from' : '2013-04-01',
            'epc_to' : '2014-03-31',
            'epc_filter_trans' : new_build_epcs,
            'validation_label' : 'apr-mar_existing'
        },
        # 2d. 12 months of EPCs, no new builds + no ECO
        {
            'place' : 'derbyshire',
            'season' : '2013-2014' ,
            'epc_from' : '2013-04-01',
            'epc_to' : '2014-03-31',
            'epc_filter_trans' : green_deal_epcs+[new_build_epcs],
            'validation_label' : 'apr-mar_existing_noECO'
        },
        # 3a. 6 months of EPCs, all
        {
            'place' : 'derbyshire',
            'season' : '2013-2014' ,
            'epc_from' : '2013-10-01',
            'epc_to' : '2014-03-31',
            'epc_filter_trans' : '' ,
            'validation_label' : 'oct-feb_all'
        },
        # 3b. 6 months of EPCs, no green deal etc
        {
            'place' : 'derbyshire',
            'season' : '2013-2014' ,
            'epc_from' : '2013-10-01',
            'epc_to' : '2014-03-31',
            'epc_filter_trans' : green_deal_epcs ,
            'validation_label' : 'oct-feb_noECO'
        },
    ]
    
    for ex in experiments:
        place,season,epc_from,epc_to,epc_filter_trans,validation_label = ex.values()

        output_dir,lsoa_file,output_epc_file = build_filenames(
            rootdir,place,season,validation_label,product_label)

        # Aggregate EPCs only if file not present
        if not os.path.exists(output_epc_file):
            print('STEP 1: aggregate EPCs')
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            lsoa_with_epc = aggregate_epcs(place,season,epc_from,epc_to,
                                           lsoa_file,output_epc_file,
                                           exclude_epc_trans=epc_filter_trans)
        else:
            print('STEP 1: aggregate EPCs geojson exists: reading it')
            lsoa_with_epc = gpd.read_file(output_epc_file)

        # Run validation on this data set
        print('STEP 2: run validation')
        run_validation(lsoa_with_epc,output_dir)
        break
    return

    

# ============================================================================
""" Processing Functions"""

def aggregate_epcs(place_label,
                   date_label,
                   epc_min_date,
                   epc_max_date,
                   lsoa_file,
                   output_epc_file,
                   exclude_epc_trans=''):
    """
    Open EPC data files and aggregate

    """

    # Get LSOA and Postcode Geodata
    print('Reading LSOA and postcode data')
    lsoa = gpd.read_file(lsoa_file) 
    pc_cent = pd.read_csv(pc_cent_file)
    
    pc_cent_gdf = pc_cent
    # The line below does not seem to be necessary... because I don't use spatial join but join on LSOA instead
    #pc_cent_gdf = df_to_gdf(pc_cent)
    
    # Assign postcode in standard format
    pc_cent_gdf = pc_cent_gdf.assign(postcode = pc_cent_gdf['pcd'].str.upper().str.replace(' ',''))

    # List postcodes associated with each LSOA OF INTEREST
    print('Listing relevant postcodes (this takes about 5 minutes)')
    lsoa_postcode_indices = []
    tic=time.process_time()
    for thislsoa in lsoa['LSOA code']:
        inds, = np.where(pc_cent_gdf['lsoa11'] == thislsoa)
        lsoa_postcode_indices = lsoa_postcode_indices + [inds]
    # this does not do list concatenation, although it looks like it
    # in fact the output is a list of np.ndarrays
    # the length of the list is the number of LSOAs
    # the length of the array in each element of the list is the number of postcodes in that LSOA

    toc = time.process_time()
    print('Done. This took {:.1f} s'.format((toc-tic)/1.0))

    # Construct EPC filenames to look for, using LAD codes from LSOA file
    print('Selecting EPC files for LADs in AoI')
    epc_files = []
    for code,name in zip(lsoa['LAD code'].unique(),
                         lsoa['LAD name'].unique()):
        epc_files += [epc_filename(code,name)]

    print('There should be {} files'.format(len(epc_files)))

    # Check existence of all EPC files, read the data, and list LAD codes
    # associated with each EPC file

    print('Finding and opening EPC files')
    epc_list = []
    notfound = []
    epc_code_list = []
            
    for epc_ind in range(len(epc_files)):
        filename = epc_files[epc_ind]
        if len(glob.glob(filename)) == 0:
            notfound += [filename]
        else:
            epc_data = pd.read_csv(filename)
            epc_data = epc_data[keepcols_epc]
            epc_list += [epc_data]
            epc_code_list += [lsoa['LAD code'].iloc[epc_ind]]
    print('{} files opened'.format(len(epc_list)))
    print('{} files not found'.format(len(notfound)))
    for ii in range(len(notfound)):
        print(notfound[ii])

    """
    We now loop over the LSOAs, find the LAD, go to the relevant EPC data
    select the postcode indices for that LSOA, use these to select the EPCs, 
    and derive relevant aggregated data on all the EPCs in that LSOA. 
    """

    # Assign new columns
    filelength = len(lsoa)
    lsoa_with_epc = lsoa.assign(N_EPC = np.zeros(filelength),
                                AveDate_EPC = np.zeros(filelength),
                                AveCEE_EPC = np.zeros(filelength),
                                AveRoofEE_EPC = np.zeros(filelength),
                                AveWallsEE_EPC = np.zeros(filelength),
                                AveWindowsEE_EPC = np.zeros(filelength),
                                FracPoorRoof_EPC = np.zeros(filelength),
                                FracPoorAll_EPC = np.zeros(filelength),
                                FracEFG_EPC = np.zeros(filelength)
                                )                                 
                                     
    # Assign column numbers
    lastcol = len(lsoa_with_epc.keys())-1
    FracEFG_EPC_col = lastcol
    FracPoorAll_EPC_col = lastcol-1
    FracPoorRoof_EPC_col = lastcol-2
    AveWindowsEE_EPC_col = lastcol-3
    AveWallsEE_EPC_col = lastcol-4
    AveRoofEE_EPC_col = lastcol-5
    AveCEE_EPC_col = lastcol-6
    AveDate_EPC_col = lastcol-7
    N_EPC_col = lastcol-8

    print('Aggregating EPCs in each LSOA (this will take a couple of minutes')
    tic=time.process_time()
    for lsoa_ind in range(filelength):
        thislsoa = lsoa.iloc[lsoa_ind]
        thislsoa_postcode_indices = lsoa_postcode_indices[lsoa_ind]
        thislsoa_postcodes = pc_cent_gdf['postcode'].iloc[thislsoa_postcode_indices]
        
        # Find relevant epc file
        thislsoa_lad_code = thislsoa['LAD code']
        try:
            thislsoa_epc_ind = np.squeeze(np.where(
                np.array(epc_code_list) == thislsoa_lad_code))[0]
        except:
            print(thislsoa_lad_code,epc_code_list,np.where(
                np.array(epc_code_list) == thislsoa_lad_code))
            raise
        epc_data = epc_list[thislsoa_epc_ind]
        epc_postcodes = epc_data['POSTCODE'].str.upper().str.replace(' ','')
        epc_data = epc_data.assign(POSTCODE_nosp = epc_postcodes)
        
        # Search postcodes 
        thislsoa_epcs = pd.merge(pd.DataFrame(thislsoa_postcodes),
                                 epc_data,
                                 left_on='postcode',
                                 right_on='POSTCODE_nosp',
                                 how='left'
                                )
        
        # Filter EPCs
        valid_trans = filter_trans_type(thislsoa_epcs,
                                        exclude_epc_trans,
                                        exclude=True)
        valid_date = filter_dates(thislsoa_epcs, 
                                  epc_min_date, 
                                  epc_max_date)
        
        valid_epc = valid_trans & valid_date

        # Aggregate data
        rows, = np.where(valid_epc)
        epc_count = len(rows) 
        epc_dates = date_to_epoch(epc_dates[rows])
        epc_cee = pd.to_numeric(
            thislsoa_epcs.iloc[rows]['CURRENT_ENERGY_EFFICIENCY'],
            errors='coerce'
        )
        epc_rfee = ee_str_to_float(thislsoa_epcs.iloc[rows]['ROOF_ENERGY_EFF'])
        epc_wlee = ee_str_to_float(thislsoa_epcs.iloc[rows]['WALLS_ENERGY_EFF'])
        epc_wdee = ee_str_to_float(thislsoa_epcs.iloc[rows]['WINDOWS_ENERGY_EFF'])
            
        lsoa_with_epc.iloc[lsoa_ind,N_EPC_col] += epc_count
        lsoa_with_epc.iloc[lsoa_ind,AveDate_EPC_col] += np.sum(epc_dates)
        lsoa_with_epc.iloc[lsoa_ind,AveCEE_EPC_col] += np.sum(epc_cee)
        lsoa_with_epc.iloc[lsoa_ind,AveRoofEE_EPC_col] += np.sum(epc_rfee)
        lsoa_with_epc.iloc[lsoa_ind,AveWallsEE_EPC_col] += np.sum(epc_wlee)
        lsoa_with_epc.iloc[lsoa_ind,AveWindowsEE_EPC_col] += np.sum(epc_wdee)
        lsoa_with_epc.iloc[lsoa_ind,FracEFG_EPC_col] += np.sum(epc_cee<=54)
        lsoa_with_epc.iloc[lsoa_ind,FracPoorRoof_EPC_col] += np.sum(epc_rfee<2.5)
        lsoa_with_epc.iloc[lsoa_ind,FracPoorAll_EPC_col] += np.sum((epc_rfee<2.5)
                                                                   & (epc_wlee<2.5)
                                                                   & (epc_wdee<2.5))

    toc = time.process_time()
    print('Done. This took {:.1f} s'.format((toc-tic)/1.0))
       
    # Having summed up all the dates and CEE's, must divide by counts to get average:
    lsoa_with_epc['AveDate_EPC'] /= lsoa_with_epc['N_EPC']
    lsoa_with_epc['AveDate_EPC'] = epoch_to_date(lsoa_with_epc['AveDate_EPC'])
    lsoa_with_epc['AveCEE_EPC'] /= lsoa_with_epc['N_EPC']
    lsoa_with_epc['AveRoofEE_EPC'] /= lsoa_with_epc['N_EPC']
    lsoa_with_epc['AveWallsEE_EPC'] /= lsoa_with_epc['N_EPC']
    lsoa_with_epc['AveWindowsEE_EPC'] /= lsoa_with_epc['N_EPC']
    # For the fractions, we have the total count of poor EPCs, so we divide by the total count of all EPCs
    lsoa_with_epc['FracEFG_EPC'] /= lsoa_with_epc['N_EPC']
    lsoa_with_epc['FracPoorRoof_EPC'] /= lsoa_with_epc['N_EPC']
    lsoa_with_epc['FracPoorAll_EPC'] /= lsoa_with_epc['N_EPC']

    # Output the results:

    # This is stupid but it won't output DateTimes correctly - have to convert to float or string
    lsoa_with_epc = lsoa_with_epc.assign(
        AveDate_EPC = lsoa_with_epc['AveDate_EPC'].astype(str))
    
    if len(glob.glob(output_epc_file))>0:
        print('Warning: overwriting existing file')
        os.system('rm -rf '+output_epc_file)

    print('Outputing data to '+output_epc_file)
    lsoa_with_epc.to_file(output_epc_file,driver='GeoJSON')

    print('EPC aggregation done')    
    return lsoa_with_epc


def run_validation(lsoa_with_epc,output_dir):  
    """ 
    Create validation outputs: plots, table of results

    """

    # First thing is to add Energy Efficiency/DTR parameters to the dataframe
    print('Fetching Energy Efficiency data')
    lsoa_all = eee.get_estimates(lsoa_with_epc)
    
    print('Making some plots')
    
    figtitle = 'All LSOAs'
    filename = output_dir+'All_LSOA_scatter.png'
    subset = range(len(lsoa_with_epc))
    mcolor = '0.5'
    validation_plot_matrix(lsoa_all[subset],filename,figtitle,mcolor=mcolor)

    figtitle = 'Urban LSOAs'
    filename = output_dir+'Urban_LSOA_scatter.png'
    subset = lsoa_all['LC_urban'].values == 1
    mcolor = 'r'
    validation_plot_matrix(lsoa_all[subset],filename,figtitle,mcolor=mcolor)

    figtitle = 'Rural LSOAs'
    filename = output_dir+'Rural_LSOA_scatter.png'
    subset = lsoa_all['LC_urban'].values == 0
    mcolor = 'k'
    validation_plot_matrix(lsoa_all[subset],filename,figtitle,mcolor=mcolor)

    return

def validation_plot_matrix(lsoa_data,filename,fig_title,mcolor='0.5'):
    """Make a big matrix of scatter plots

    Inputs: 
    lsoa_data = dataframe 
    filename = path to save figure to
    fig_title = string title of figure

    Rows: 
    LST
    DTR1
    DTR2
    DTR3

    Cols:
    EPC CEE
    Roof EE
    Comb EE
    Frac EFG
    Frac Poor Roof
    Frac Poor All
    """
    
    rowdata = [{'ydata':lsoa_data['LST_mean'].values, 'ylabel':'r LST','ylim':[-2,2]},
               {'ydata':lsoa_data['DTR1'].values, 'ylabel':'DTR1','ylim':[5,70]},
               {'ydata':lsoa_data['LST2'].values, 'ylabel':'DTR2','ylim':[5,70]},
               {'ydata':lsoa_data['LST3'].values, 'ylabel':'DTR3','ylim':[5,70]}]

    coldata = [{'xdata':lsoa_data['AveCEE_EPC'].values, 'xlabel':'Ave CEE', 'xlim':[40,90], 'expect':'-'},
               {'xdata':lsoa_data['AveRoofEE_EPC'].values, 'xlabel':'Ave Roof EE', 'xlim':[0.5,5.5], 'expect':'-'},
               {'xdata':lsoa_data['AveCombEE_EPC'].values, 'xlabel':'Ave Comb EE', 'xlim':[4,16], 'expect':'-'},
               {'xdata':lsoa_data['FracEFG_EPC'].values*100, 'xlabel':'% EFG', 'xlim':[-5,90], 'expect':'+'},
               {'xdata':lsoa_data['FracPoorRoof_EPC'].values*100, 'xlabel':'% Poor Roof', 'xlim':[-5,90], 'expect':'+'},
               {'xdata':lsoa_data['FracPoorAll_EPC'].values*100, 'xlabel':'% Poor All', 'xlim':[-2,30], 'expect':'+'}]

    ncols=len(coldata)
    nrows=len(rowdata)

    fig,axes = plt.subplots(nrows, ncols, figsize=(2*ncols,2*nrows))
    fig.subplots_adjust(left=0.05,right=0.95,
                            bottom=0.05,top=0.9,
                            wspace=0.,hspace=0.) 

    fig.suptitle(fig_title,size=15)

    for jj in range(nrows):
        for ii in range(ncols):
            ax=axes[jj,ii]
            
            valid, = np.where(np.isfinite(coldata[ii]['xdata']) & np.isfinite(rowdata[jj]['ydata']))
            xdata = coldata[ii]['xdata'][valid]
            ydata = rowdata[jj]['ydata'][valid]
            
            # Plot data
            ax.scatter(coldata[ii]['xdata'],rowdata[jj]['ydata'],marker='.',color=mcolor,alpha=0.20)
            
            # Show correlation
            spr,spp = spearmanr(xdata,ydata)
            ax.text(0.1,0.9,r'$\rho$= {:.2F}, p={:.1e}'.format(spr,spp),
                    color='r',transform=ax.transAxes)
            
            # Show regression
            slope,intercept,r,pval,stderr = linregress(xdata,ydata)
            #print(ii,jj,slope,intercept,r**2,pval,stderr)
            xfit = np.array(coldata[ii]['xlim'])
            yfit = xfit*slope + intercept
            ax.plot(xfit,yfit,color='b',lw=0.5)
            ax.text(0.1,0.02,'m= {:.2F}, p={:.1e} \nR$^2$={:.6F}'.format(slope,pval,r**2),
                    color='b',transform=ax.transAxes)
            
            # Show expectation
            if jj==0:
                ax.set_title('expect: {}'.format(coldata[ii]['expect']))
                
            # Axes 
            ax.set_xlim(coldata[ii]['xlim'])
            ax.set_ylim(rowdata[jj]['ylim'])
            
            if ii==0:
                ax.set_ylabel(rowdata[jj]['ylabel'])
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
                
            if jj==nrows-1:
                ax.set_xlabel(coldata[ii]['xlabel'])
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            
    fig.savefig(filename)
    return 


# ============================================================================
""" Utility Functions"""

def build_filenames(rootdir,place,season,validation_label,product_label):
    """ Tell the main function where to locate geojson files"""

    output_dir = rootdir+'output_LSOA_LST/{}_{}/validation_EPC_{}/'.format(
        place,season,validation_label)
    lsoa_file = rootdir+'output_LSOA_LST/{0}_{1}/lsoa_{0}_{2}_LC.geojson'.format(
        place,season,product_label)
    output_epc_file = output_dir+'lsoa_{}_{}_LC_EPC.geojson'.format(
        place,product_label)

    return output_dir,lsoa_file,output_epc_file

def df_to_gdf(df):
    """
    Convert pandas dataframe with x,y columns to geodataframe with geometry
    """

    from shapely.geometry import Point

    pcx = pd.to_numeric(df['X'],errors='coerce')
    pcy = pd.to_numeric(df['Y'],errors='coerce')

    pc_geom = list(zip(pcx,pcy))
    pc_geom = [Point(xy) for xy in pc_geom]

    gdf = gpd.GeoDataFrame(df,geometry=pc_geom)
    return gdf

def epc_filename(LAD_code,LAD_name):
    """Build EPC filename"""

    filename = raw_data_dir \
        +'attributes/epc_lad_england_wales/' \
        +'-'.join(['domestic',LAD_code]
                  +LAD_name.replace(',','').replace("'",'-').replace('.','').split()
                 ) \
        +'/certificates.csv'
    return filename

def filter_trans_type(df,values,exclude=False):
    """
    Return a boolean vector =True where df['Transaction Type'] is in values 

    if exclude=True, return True where df['Transaction Type'] is not in values 

    Possible values of Transaction Type:
    assessment for green deal
    ECO assessment
    FiT application
    following green deal
    marketed sale
    new dwelling
    NO DATA!
    non marketed sale
    none of the above
    not recorded
    rental
    rental (private)
    rental (private) - this is for backwards compatibility...
    rental (social)
    RHI application
    unknown

    see: https://www.ofgem.gov.uk/environmental-programmes
    """

    series = df['Transaction Type']

    if np.size(values)==1:
        if exclude:
            bool_arr = ~series.str.match(values,case=False).values
        else:
            bool_arr = series.str.match(values,case=False).values

    elif np.size(values)>1:
        bool_arr = np.array([False]*len(df))
        if exclude:
            for item in values:
                bool_arr = bool_arr | series.str.match(item,case=False).values
        else:
            for item in values:
                bool_arr = bool_arr & ~series.str.match(item,case=False).values

    else:
        bool_arr = [True]*len(series)

    return (bool_arr==True)

def filter_dates(df, min_date, max_date):
    """ Return a boolean vector =True where df['INSPECTION_DATE'] in range"""

    epc_dates = pd.to_datetime(df['INSPECTION_DATE'],errors='coerce',yearfirst=True)
    bool_arr = (epc_dates <= pd.Timestamp(max_date)) & (epc_dates >= pd.Timestamp(min_date))
    return 

def ee_str_to_float(ee_as_str_series):
    """
    Convert EPC energy efficiency ratings to floats

    input: epc_as_str_series = pd.Series of strings 'VERY POOR' - 'VERY GOOD'
    return: pd.Series of floats 1.0 - 5.0
    """

    str_series = ee_as_str_series.str.upper().copy()
    str_series = str_series.str.replace('VERY GOOD','5.0')
    str_series = str_series.str.replace('GOOD','4.0')
    str_series = str_series.str.replace('AVERAGE','3.0')
    str_series = str_series.str.replace('VERY POOR','1.0')
    str_series = str_series.str.replace('POOR','2.0')
    str_series = str_series.str.replace('NAN','')

    return pd.to_numeric(str_series,errors='coerce',downcast='float')

def date_to_epoch(date_series, unit='1d', epoch='1970-01-01'):
    return (date_series - pd.Timestamp(epoch)) // pd.Timedelta(unit)

def epoch_to_date(numeric_series, unit='D'): 
    return pd.to_datetime(numeric_series.values, errors='coerce', unit=unit)



if __name__ == "__main__":
    main()