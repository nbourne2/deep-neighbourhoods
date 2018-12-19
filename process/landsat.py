"""
Methods for querying landsat data on AWS and Google Cloud Services

Requirements: 
pandas
boto3 (for AWS)
google-cloud-storage
shapely

Usage:
# Access AWS
aws = landsat.aws()

# Find scenes in csv list matching some criteria
tab = aws.read_scene_list()
rows = aws.find_in_scene_list(tab,path=203,row=23,maxDate='20170101')
sort = np.argsort(tab['acquisitionDate'][rows].values)
print(len(sort))
print(tab[rows].iloc[sort])

# Find scenes in bucket matching criteria
scenes,dates = aws.find_in_bucket(203,23,'QA',maxDate='20170101')
sort = np.argsort(dates)
print(len(sort))
print(scenes[sort])

# Open in rasterio over https
with rasterio.open(aws.https_stub+scenes[sort][0]) as src:
    arr = src.read(out_shape=(src.height//100, src.width//100))
    print('success')

# Open in rasterio on S3 using default profile in ~/.aws/credentials
import rasterio
with rasterio.Env(session=rasterio.session.AWSSession(profile_name='default')):
    with rasterio.open(aws.s3_stub+scenes[sort][0]) as src:
        arr = src.read(out_shape=(src.height//100, src.width//100))
        print('success')


# Access Google
gcs = landsat.gcs()

# Find scenes in csv list matching some criteria
tab = gcs.read_scene_list()
rows = gcs.find_in_scene_list(tab,path=203,row=23,maxDate='20170101',Collection=1)
sort = np.argsort(tab['DATE_ACQUIRED'][rows].values)
print(len(sort))
print(tab[rows].iloc[sort])

# Find scenes in bucket matching criteria
scenes,dates = gcs.find_in_bucket(203,23,'QA',maxDate='20170101')
sort = np.argsort(dates)
print(len(sort))
print(scenes[sort])

# Open in rasterio over https
with rasterio.open(gcs.https_stub+myscenes[sort][0]) as src:
    arr = src.read(out_shape=(src.height//100, src.width//100))
    print('success')

# Open in rasterio and read window: (AoIbounds must be defined relative to src.crs)
with rasterio.open(gcs.https_stub+myscenes[sort][0]) as src:
    AoIrow0,AoIcol0 = rasterio.transform.rowcol(src.transform,AoIbounds['minx'],AoIbounds['maxy'])
    AoIrow1,AoIcol1 = rasterio.transform.rowcol(src.transform,AoIbounds['maxx'],AoIbounds['miny'])
    AoIheight,AoIwidth = AoIrow1 - AoIrow0, AoIcol1 - AoIcol0
    window = rasterio.windows.Window(AoIcol0,AoIrow0,AoIwidth,AoIheight)
    arr = src.read(window=window)
    print('success')

"""

import re
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

class aws():
    """ Interact with Landsat on AWS"""

    def __init__(self):
        self.bucket_name = 'landsat-pds'
        self.https_stub = 'https://'+self.bucket_name+'.s3.amazonaws.com/'
        self.s3_stub = 's3://'+self.bucket_name+'/'
        
    def read_scene_list(self):
        """Return pandas dataframe
        Columns: 'productId', 'entityId', 'acquisitionDate', 'cloudCover',
       'processingLevel', 'path', 'row', 'min_lat', 'min_lon', 'max_lat',
       'max_lon', 'download_url'
        """
        return pd.read_csv(self.https_stub+'c1/L8/scene_list.gz')

    def find_in_scene_list(self,scene_list,
                           path=None,row=None,
                           minDate=None,maxDate=None,
                           maxCloudCoverPc=None,
                           procLevel=None,
                           AoI = None
                          ):
        """Find scenes matching argument criteria in the scene list

        Warning: do not assume all scenes are necessarily in the list
        Also note that the list only contains scenes in Collection

        Inputs (all optional):
        path, row = WRS path,rowm if known
        minDate,maxDate = can specify either,both, or neither - has to be interpretable by pd.datetime
        maxCloudCoverPc = percentage
        proclevel = L1TP, eg.
        AoI = dict containing minx,maxx,miny,maxy in lon,lat units(?)

        Returns: 
        Array that is True for rows matching all criteria

        """

        nrows = len(scene_list)
        m_path = np.array([True]*nrows)
        m_row = np.array([True]*nrows)
        m_date1 = np.array([True]*nrows)
        m_date2 = np.array([True]*nrows)
        m_cloud = np.array([True]*nrows)
        m_proc = np.array([True]*nrows)
        m_aoi = np.array([True]*nrows)

        if path is not None:
            m_path = scene_list['path'].values.astype(np.int) == np.int(path)
        if row is not None:
            m_row = scene_list['row'].values.astype(np.int) == np.int(row)
        if minDate is not None:
            m_date1 = pd.to_datetime(scene_list['acquisitionDate']) >= pd.to_datetime(minDate)
        if maxDate is not None:
            m_date2 = pd.to_datetime(scene_list['acquisitionDate']) <= pd.to_datetime(maxDate)
        if maxCloudCoverPc is not None:
            m_cloud = scene_list['cloudCover'].values.astype(np.float) <= np.float(maxCloudCoverPc)
        if procLevel is not None:
            m_proc = scene_list['processingLevel'].values.astype(np.str) == np.str(procLevel)
        if AoI is not None:
            if type(AoI)==dict:
                m_aoi = []
                ax0,ax1,ay0,ay1 = AoI['minx'],AoI['maxx'],AoI['miny'],AoI['maxy']
                # TO DO: replace this awful loop with something using geopandas series
                # something like https://geopandas.readthedocs.io/en/latest/gallery/create_geopandas_from_pandas.html 
                for s in scene_list:
                    sx0,sx1 = s['min_lon'],s['max_lon']
                    sy0,sy1 = s['min_lat'],s['max_lat']
                    p1 = Polygon([(sx0,sy0), (sx0,sy1), (sx1,sy1), (sx1,sy0)])
                    p2 = Polygon([(ax0,ay0), (ax0,ay1), (ax1,ay1), (ax1,ay0)])
                    m_aoi += [p1.intersects(p2)]
                # m_aoi = ((scene_list['max_lon']<AoI['minx'] |
                #          scene_list['min_lon']>AoI['maxx']) & 
                #          (scene_list['max_lat']<AoI['miny'] |
                #          scene_list['min_lat']>AoI['maxy']))

        return (m_path & m_row & m_date1 & m_date2 & m_cloud & m_proc & m_aoi)
    
    def find_in_bucket(self,path,row,band,Collection=True,
                       minDate=pd.Timestamp.min,maxDate=pd.Timestamp.max):
        """
        Look for Landsat-8 scenes matching these criteria in the bucket

        Inputs:
        path = WRS path (int or str)
        row = WRS row
        band = int or str: Note that a band must be specified (if it was not coded this way, it would match all bands)
        Collection: specify whether to look in Collection or Pre-collection (one or the other, not both)
        minDate,maxDate = [optional] anything that can be parsed by pd.to_datetime()

        Returns: tuple (list of scene names, list of pd.datetimes of acquisitions)
        """
        import boto3

        bucket = boto3.resource('s3').Bucket(self.bucket_name)

        if collection:
            prefix = f'c1/L8/{path}/{row}/'
            search_str = (
                r'L(\w{1})08_(\w{2})(\w{2})_(\d{3})(\d{3})_(\d{8})_(\d{8})_(\d{2})_(\w{2})/[\w]+B'
                + str(band)
                + r'\.TIF$'
                )
        else:
            prefix = f'L8/{path}/{row}/'
            search_str = (
                r'L(\w{1,2})8(\d{3})(\d{3})(\d{4})(\d{3})(\w{3})(\w{2})/.+B'
                + str(band)
                + r'\.TIF$'
                )
            
        good_scenes=[]
        good_scene_dates=[]
        # The following line will list scenes matching the prefix:
        for objsum in bucket.objects.filter(Prefix=prefix,MaxKeys=999999): #,Delimiter='/'
            obs_meta = re.match(prefix+search_str, objsum.key)
            if obs_meta is None:
                continue
            obs_meta = obs_meta.groups()

            if collection:
                obs_sens,obs_lev,obs_proc,obs_path,obs_row,obs_date,pro_date,coll_no,coll_cat = obs_meta
                Tobs = pd.to_datetime(obs_date)
            else:
                obs_sens,obs_path,obs_row,obs_year,obs_jday,obs_gsi,obs_ver = obs_meta
                Tobs = pd.to_datetime(obs_year+obs_jday,format='%Y%j')

            if ((Tobs>=pd.to_datetime(minDate)) and 
                (Tobs<=pd.to_datetime(maxDate))):
                    good_scenes += [objsum.key]
                    good_scene_dates += [Tobs]
        
        return np.array(good_scenes),np.array(good_scene_dates)



class gcs():
    """ Interact with Landsat on Google Cloud Storage"""

    def __init__(self):
        self.bucket_name = 'gcp-public-data-landsat'
        self.https_stub = 'https://storage.googleapis.com/'+self.bucket_name+'/'
        self.gs_stub = 'gs://'+self.bucket_name+'/'
        
    def read_scene_list(self):
        """Return pandas dataframe
        Columns: 'SCENE_ID', 'PRODUCT_ID', 'SPACECRAFT_ID', 'SENSOR_ID', 'DATE_ACQUIRED',
       'COLLECTION_NUMBER', 'COLLECTION_CATEGORY', 'SENSING_TIME', 'DATA_TYPE',
       'WRS_PATH', 'WRS_ROW', 'CLOUD_COVER', 'NORTH_LAT', 'SOUTH_LAT',
       'WEST_LON', 'EAST_LON', 'TOTAL_SIZE', 'BASE_URL'
        """
        return pd.read_csv(self.https_stub+'index.csv.gz')

    def find_in_scene_list(self,scene_list,
                           path=None,row=None,
                           minDate=None,maxDate=None,
                           maxCloudCoverPc=None,
                           procLevel=None,
                           AoI = None,
                           Collection = None,
                           Category = None,
                           Spacecraft = 'LANDSAT_8',
                           Sensor = 'OLI_TIRS'
                          ):
        """Find scenes matching argument criteria in the scene list

        Warning: do not assume all scenes are necessarily in the list
        
        Inputs (all optional):
        path, row = WRS path,rowm if known
        minDate,maxDate = can specify either,both, or neither - has to be interpretable by pd.datetime
        maxCloudCoverPc = percentage
        proclevel = L1TP, eg.
        AoI = dict containing minx,maxx,miny,maxy in lon,lat units(?)
        Collection = '1',1,True: Collection-1; 
                      anything else: PRE; 
                      None: both.
        Category = 'RT', 'T1', 'T2' (for Collection-1 only)

        Returns: 
        Array that is True for rows matching all criteria

        """

        nrows = len(scene_list)
        m_path = np.array([True]*nrows)
        m_row = np.array([True]*nrows)
        m_date1 = np.array([True]*nrows)
        m_date2 = np.array([True]*nrows)
        m_cloud = np.array([True]*nrows)
        m_proc = np.array([True]*nrows)
        m_aoi =  np.array([True]*nrows)
        m_col = np.array([True]*nrows)
        m_cat = np.array([True]*nrows)
        m_sat = np.array([True]*nrows)
        m_sens = np.array([True]*nrows)

        if path is not None:
            m_path = scene_list['WRS_PATH'].values.astype(np.int) == np.int(path)
        if row is not None:
            m_row = scene_list['WRS_ROW'].values.astype(np.int) == np.int(row)
        if minDate is not None:
            m_date1 = pd.to_datetime(scene_list['DATE_ACQUIRED']) >= pd.to_datetime(minDate)
        if maxDate is not None:
            m_date2 = pd.to_datetime(scene_list['DATE_ACQUIRED']) <= pd.to_datetime(maxDate)
        if maxCloudCoverPc is not None:
            m_cloud = scene_list['CLOUD_COVER'].values.astype(np.float) <= np.float(maxCloudCoverPc)
        if procLevel is not None:
            m_proc = scene_list['DATA_TYPE'].values.astype(np.str) == np.str(procLevel)
        if AoI is not None:
            if type(AoI)==dict:
                m_aoi = []
                ax0,ax1,ay0,ay1 = AoI['minx'],AoI['maxx'],AoI['miny'],AoI['maxy']
                # TO DO: replace this awful loop with something using geopandas series
                # something like https://geopandas.readthedocs.io/en/latest/gallery/create_geopandas_from_pandas.html 
                for s in scene_list:
                    sx0,sx1 = s['WEST_LON'],s['EAST_LON']
                    sy0,sy1 = s['SOUTH_LAT'],s['NORTH_LAT']
                    p1 = Polygon([(sx0,sy0), (sx0,sy1), (sx1,sy1), (sx1,sy0)])
                    p2 = Polygon([(ax0,ay0), (ax0,ay1), (ax1,ay1), (ax1,ay0)])
                    m_aoi += [p1.intersects(p2)]
                # m_aoi = ((scene_list['EAST_LON']<AoI['minx'] |
                #          scene_list['WEST_LON']>AoI['maxx']) & 
                #          (scene_list['NORTH_LAT']<AoI['miny'] |
                #          scene_list['SOUTH_LAT']>AoI['maxy']))
        if Collection is not None:
            if int(Collection)==1:
                m_col = scene_list['COLLECTION_NUMBER'].values.astype(np.str) == '1'
            else: 
                m_col = scene_list['COLLECTION_NUMBER'].values.astype(np.str) == 'PRE'
        if Category is not None:
            m_cat = scene_list['COLLECTION_CATEGORY'].values.astype(np.str) == Category
        if Spacecraft is not None:
            m_sat = scene_list['SPACECRAFT_ID'].values.astype(np.str) == Spacecraft
        if Sensor is not None:
            m_sens = scene_list['SENSOR_ID'].values.astype(np.str) == Sensor
            
        return (m_path & m_row & m_date1 & m_date2 & m_cloud & m_proc & m_aoi & m_col & m_cat & m_sat & m_sens)

    def find_in_bucket(self,path,row,band,Collection=True,
                       minDate=pd.Timestamp.min,maxDate=pd.Timestamp.max):
        """
        Look for Landsat-8 scenes matching these criteria in the bucket

        Inputs:
        path = WRS path (int or str)
        row = WRS row
        band = int or str: Note that a band must be specified (if it was not coded this way, it would match all bands)
        Collection: specify whether to look in Collection or Pre-collection (one or the other, not both)
        minDate,maxDate = [optional] anything that can be parsed by pd.to_datetime()

        Returns: tuple (list of scene names, list of pd.datetimes of acquisitions)
        """
        from google.cloud import storage

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(self.bucket_name)

        if Collection:
            prefix = f'LC08/01/{path}/{row}/' 
            search_str = (
                r'L(\w{1})08_(\w{2})(\w{2})_(\d{3})(\d{3})_(\d{8})_(\d{8})_(\d{2})_(\w{2})/[\w]+B'
                + str(band)
                + r'\.TIF$'
                )
        else:
            prefix = f'LC08/PRE/{path}/{row}/'
            search_str = (
                r'L(\w{1,2})8(\d{3})(\d{3})(\d{4})(\d{3})(\w{3})(\w{2})/.+B'
                + str(band)
                + r'\.TIF$'
                )
            
        good_scenes=[]
        good_scene_dates=[]
        # The following line will list scenes matching the prefix:
        for blob in bucket.list_blobs(prefix=prefix): #,delimiter='/'
            obs_meta = re.match(prefix+search_str, blob.name)
            if obs_meta is None:
                continue
            obs_meta = obs_meta.groups()
            if Collection:
                obs_sens,obs_lev,obs_proc,obs_path,obs_row,obs_date,pro_date,coll_no,coll_cat = obs_meta
                Tobs = pd.to_datetime(obs_date)
            else:
                obs_sens,obs_path,obs_row,obs_year,obs_jday,obs_gsi,obs_ver = obs_meta
                Tobs = pd.to_datetime(obs_year+obs_jday,format='%Y%j')
            if ((Tobs>=pd.to_datetime(minDate)) and 
                (Tobs<=pd.to_datetime(maxDate))):
                    good_scenes += [blob.name]
                    good_scene_dates += [Tobs]
        
        return np.array(good_scenes),np.array(good_scene_dates)
