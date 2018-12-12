""" Module for accessing Met Office climate data "UKCP09" on air temperature

Usage:
at = met_climate.access_ukcp09(username,password)
month = 1 # January
with rasterio.open(scene_filename) as src: 
    at_transform = src.transform
    at_data = at.grid_temp_over_scene(src, month)
ax = rplt.show(at_data,transform=at_transform)

For more info see method access_ukcp09.grid_temp_over_scene()

Data sources: 
Long term averages by month, gridded at 5km res:
http://catalogue.ceda.ac.uk/uuid/620f6ed379d543098be1126769111007

Daily data gridded at 5km res:
http://catalogue.ceda.ac.uk/uuid/319b3f878c7d4cbfbdb356e19d8061d6

See also "MIDAS" dataset for up-to-date weather station data (not gridded)
http://catalogue.ceda.ac.uk/uuid/220a65615218d5c9cc9e4785a3234bd0

The full gridded Data Collection is described here:
http://catalogue.ceda.ac.uk/uuid/87f43af9d02e42f483351d79b3d6162a
Access to all data:
http://data.ceda.ac.uk/badc/ukcp09/data/gridded-land-obs/
Publication:
https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/joc.1161

A username and password must be supplied to access the data
Usage of the data is subject to license:
http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

Info on the NetCDF data format 
https://help.ceda.ac.uk/article/106-netcdf
https://www.unidata.ucar.edu/software/netcdf/docs/

"""
import numpy as np
import pandas as pd
import rasterio as rst
from rasterio.warp import reproject, Resampling, calculate_default_transform
from scipy.interpolate import griddata
from pydap.client import open_url
from pydap.cas.get_cookies import setup_session

class access_ukcp09():
    def __init__(self,username,password):
        self.url = 'http://data.ceda.ac.uk/badc/ukcp09/data/gridded-land-obs/'
        testurl = self.url + \
            'gridded-land-obs-averages-5km/00README_catalogue_and_licence.txt'
        try:
            self.session = setup_session(testurl,username,password)
        except Exception as e:
            print('Error: {}'.format(e))
            return

        self.data = None
        self.xcol = 'lon'
        self.ycol = 'lat'
        self.tcol = ''
        self.crs = 'EPSG:4326'
        return

    def get_daily_meantemp(self,datestring):
        """ Return daily mean temperature of the day in question

        Input:  datestring = YYYYMMDD (string)
        Return: average of daily max and min temps in deg C (grid of float)
        """
        try:
            dayofyear = pd.to_datetime(datestring).dayofyear
            YYYY = datestring[0:4]
            date_range = '{0}0101-{0}1231'.format(YYYY)
        except:
            print('Error: invalid date string: {} should be "YYYYMMDD"'.format(
                datestring))
            return 

        dataset_url = self.url \
            +'gridded-land-obs-daily/grid/netcdf/mean-temperature/' \
            +'ukcp09_gridded-land-obs-daily_5km_mean-temperature_' \
            +'{}.nc'.format(date_range)

        # Open the OpenDAP server and define the column of interest
        try:
            self.data = open_url(dataset_url, session=self.session)
        except:
            print('Error accessing data: {}'.format(dataset_url))
            return

        # Download the actual data
        self.tcol = 'daily_meantemp'
        at_temp_grid = self.data[self.tcol][dayofyear,:,:]
        
        # Store as numpy array
        at_temp = at_temp_grid.array.data
        
        return at_temp

    def get_monthly_meantemp(self,month):
        """ Return average of daily mean temperature of all days in month
        averaged over a 30 year period

        Input (optional) month 1-12 (int) If None, return data for all 12 months
        Return: array of mean temperature in each of the 12 months, as a grid
        with dimensions (month, lon, lat)
        """
        if month is None:
            MM = range(12)+1
        else:
            try:
                MM = int(month)
                assert (MM>=1) and (MM <=12)
            except:
                print('Error: invalid month: {}'.format(month) 
                      +' should be parseable as int 1-12')
                return 
        date_range = '198101-201012'
        dataset_url = self.url \
            +'gridded-land-obs-averages-5km/grid/netcdf/mean-temperature/' \
            +'ukcp09_gridded-land-obs-averages-5km_mean-temperature_' \
            +'{}.nc'.format(date_range)

        # Open the OpenDAP server
        try:
            self.data = open_url(dataset_url, session=self.session)
        except:
            print('Error accessing data: {}'.format(dataset_url))
            return

        # Download the actual data
        self.tcol = 'monthly_meantemp'
        at_temp_grid = self.data[self.tcol][MM-1,:,:]
        
        # Store as numpy array
        at_temp = at_temp_grid.array.data
        
        return at_temp

    def grid_temp_over_scene(self, scene, date, interpolation='nearest'):
        """Obtain air temperature data covering a given (landsat) scene
        interpolated to the raster grid

        Inputs:
        scene with any CRS (if not EPSG:4326, it will be reprojected) - this 
            should be a rasterio dataset returned by rasterio.open()
        date = either a string YYYYMMDD, or an int 1-12.
            if YYYYMMDD, then find the mean daily temp on that day
            if 1-12, then find the mean temp for this month over 30-year span
        interpolation = method for interpolating with scipy.griddata
            either of ['linear','nearest','cubic'] 

        """

        # Ensure same sampling method used for scipy.griddata and 
        # rasterio.warp.reproject
        if interpolation=='nearest':
            rst_resampling = Resampling.nearest
        elif interpolation=='linear':
            rst_resampling = Resampling.bilinear
        elif interpolation=='cubic':
            rst_resampling = Resampling.bicubic
        else:
            print('Error: interpolation method not recognised: ' 
                  +'{}'.format(interpolation)
                  +' should be one of "linear","nearest","cubic"')
            return
        
        # Get air temp data and associated coordinates
        if type(date) is str:
            at_temp = self.get_daily_meantemp(date)
        elif type(date) is int:
            at_temp = self.get_monthly_meantemp(date)
        else:
            print('Error: did not recognise data type of {}'.format(date)
                  +' - expected int or str')
            return

        xdata = self.data[self.xcol][:,:].array.data.flatten()
        ydata = self.data[self.ycol][:,:].array.data.flatten()
        tdata = at_temp.flatten() 
        
        gooddata = tdata< 1e20 
        # excluding null data - otherwise there will be high temp at coasts
        xdata1 = xdata[gooddata]
        ydata1 = ydata[gooddata]
        tdata1 = tdata[gooddata]

        # Reproject raster scene onto appropriate CRS
        at_crs = self.crs
        scene_transform = scene.transform
        scene_crs = scene.crs
        at_transform, width, height = calculate_default_transform(
            scene_crs, at_crs, scene.width, scene.height, *scene.bounds)
        
        raster_shape = (scene.height, scene.width)
        
        # Interpolate air temp  data onto raster grid
        
        # First we define the grid of pixels in the SCENE and convert to xy 
        # in the SCENE coordinates
        grid_r, grid_c = np.mgrid[0:raster_shape[0], 0:raster_shape[1]].astype(
            np.float)
        sgrid_x, sgrid_y = rst.transform.xy(scene_transform,grid_r,grid_c)
        sgrid_x = np.array(sgrid_x)
        sgrid_y = np.array(sgrid_y)

        # Next we must transform those xy coordinates into the CRS of the AT
        from rasterio.warp import transform
        agrid_x, agrid_y = transform(scene_crs,at_crs,
                                     sgrid_x.flatten(),sgrid_y.flatten())

        raster_grid_xy = (np.array(agrid_x).reshape(sgrid_x.shape),
                          np.array(agrid_y).reshape(sgrid_y.shape))
        
        # Finally we interpolate the AT data onto the defined pixel grid
        at_regridded2 = griddata(np.transpose(np.array([xdata1,ydata1])),
                                 tdata1,
                                 raster_grid_xy,
                                 method=interpolation)
        
        return at_regridded2
