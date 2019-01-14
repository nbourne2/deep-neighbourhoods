
import sys
sys.path.append(
    '/home/io/ASTROSAT/code/urban_extraction/classification_module')

from constants import DATA_PATH
from indices.NDVI import NDVI
import os
import glob
from multiprocessing import Pool, cpu_count

import rasterio
from rasterio import features
import numpy as np

import pandas as pd
import geopandas as gpd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

# disable warnings
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Warning used to notify implicit data conversions happening in the code.
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
np.seterr(divide='ignore', invalid='ignore')


def load_raster(input_file):
    """Returns a raster array which consists of its bands and transformation matrix

    parameters
    ----------
    input_file: path directory to the raster file
    """
    with rasterio.open(input_file) as src:
        band_optical = src.read()
        transform = src.transform
        shape = src.shape
        profile = src.profile
        raster_img = np.rollaxis(band_optical,0,3)
        return {'band_optical': band_optical,'raster_img':raster_img ,'transform': transform, 'shape': shape, 'profile': profile}


def write_raster(raster_profile, raster_input, raster_output):
    """Write a raster file to a directory

    parameters
    ----------
    raster_profile: transformation matrix of the raster
    raster_input: raster file with dimensions (rows,columns)
    raster_output: provide a location in teh directory to write the raster
    """
    profile = raster_profile['profile']
    profile.update(
        dtype=raster_input.dtype,
        count=1,
        compress='lzw')

    with rasterio.open(raster_output, 'w', **profile) as out:
        out.write_band(1, raster_input)

def rasterize(vector):
    """Returns an image array with input geometries burned in

    parameters
    ----------
    vector: vector geometries
    """
    raster = load_raster(raster_input_data)
    labeled_pixels = np.zeros((raster['shape'][0],raster['shape'][1]))
    for i,shp in sorted(enumerate(vector)):
        label = i+1  
        df = gpd.read_file(shp)
        geom = df['geometry']
        vectors_rasterized = features.rasterize(geom,
                                            out_shape = raster['shape'],
                                            transform = raster['transform'],
                                            all_touched=True,
                                            fill=0, default_value=label)
        labeled_pixels += vectors_rasterized

    return labeled_pixels

def split(split=0.30):
    """Splits the observations into training and testing.

    parameters
    ----------
    test_size: Proportion of the data for testing. Default value is 30%.
    """
    raster = load_raster(raster_input_data)
    raster_img = raster['raster_img'] 
    labeled_pixels = rasterize(shapefiles)  
    for i, shp in enumerate(shapefiles):
        i = i+1
        shp_path = os.path.split(shp)
        land_classes = shp_path[1][:-4]
        print('Class {land_classes} contains {n} pixels'.format(land_classes=land_classes, n=(labeled_pixels == i).sum()))

    roi_int = labeled_pixels.astype(int)
    # X is the matrix containing our features
    X = raster_img[roi_int > 0] 
    # y contains the values of our training data
    y = labeled_pixels[labeled_pixels>0]

    #Split our dataset into training and testing. Test data will be used to make predictions
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify = y)

    return X_train, X_test, y_train, y_test


def tune(search_type):
    """Returns the best parameters (C and gamma) for the SVM model

    parameters
    ----------
    search_type: string
        search_type options: random, which uses RandomizedSearchCV to tune the hyperparameters
                             grid, which uses GridSearchCV to tune the hyperparameters (computational expensive)
    """
    X_train, X_test, y_train, y_test = split()
    param_range_c = np.logspace(0, 2, 8)
    param_range_gamma = np.logspace(-6, -1, 8)

    param_grid = {'svm__C': param_range_c,
                  'svm__gamma': param_range_gamma}

    pip = Pipeline([('scale', preprocessing.StandardScaler()),
                    ('svm', SVC(kernel='rbf', class_weight='balanced'))])

    if search_type == 'grid':
        clf = GridSearchCV(estimator=pip,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=3,
                           n_jobs=-1)

        clf = clf.fit(X_train, y_train)

        # print accuracy of the model
        print('Best parameters:', clf.best_params_)
        print('Classification accuracy', clf.best_score_)

    elif search_type == 'random':
        clf = RandomizedSearchCV(estimator=pip,
                                 param_distributions=param_grid,
                                 scoring='accuracy',
                                 cv=3,
                                 n_iter=15,
                                 error_score='numeric',  # it supresses the warning error
                                 n_jobs=-1)

        clf = clf.fit(X_train, y_train)

        # print accuracy of the model
        print('Best parameters:', clf.best_params_)
        print('Classification accuracy:', clf.best_score_)

    return clf

def predict(input_data):
    """It makes predictions on an array of dimensions (cols,rows,features)

    parameters
    ----------
    input_data: An array of dimentions (cols,rows,features)
    """
    y_predict = clf.predict(input_data)
    return y_predict


def parallel_processing():
    """It performs SVM classification using parallel processing

    parameters
    ----------
    no parameters are required
    """
    raster = load_raster(raster_input_data)
    raster_img = raster['raster_img'] 
    # split good data into chunks for parallel processing
    cpu_n = cpu_count()
    # Reshape the data so that we make predictions for the whole raster
    new_shape = (raster_img.shape[0] *
                 raster_img.shape[1], raster_img.shape[2])

    
    img_as_array = raster_img[:, :].reshape(new_shape)
    image_array = np.copy(img_as_array)
    split = np.array_split(image_array, cpu_n)

    # run parallel processing of all data with SVM
    pool = Pool(cpu_n)
    svmLablesPredict = pool.map(predict, split)
    # join results back from the queue and insert into full matrix
    svmLablesPredict = np.hstack(svmLablesPredict)
    svm_reshape = svmLablesPredict.reshape(
        raster_img.shape[0], raster_img.shape[1])

    return svm_reshape


def urban_extraction(input_data):
    """It returns the final array of the urban environment extraction with vegetation masked out

    parameters
    ----------
    input_data: path directory to the raster file (optical data)
    """
    svm_classified = parallel_processing()
    # use NDVI to remove vetetation
    raster = load_raster(input_data)
    red = raster['band_optical'][2, :, :]
    nir = raster['band_optical'][5, :, :]
    ndvi = NDVI(nir, red)
    svm_classified = svm_classified * ndvi

    return svm_classified

def model_accuracy():
    """It produces a classification report and confusion matrix of the classified raster

    parameters
    ----------
    no parameters are required
    """
    svm_classified = parallel_processing()
    labeled_pixels = rasterize(shapefiles)
    target_names = [os.path.split(s)[1][:-4] for s in shapefiles]

    for_verification = np.nonzero(labeled_pixels)
    verification_labels = labeled_pixels[for_verification]
    predicted_labels = svm_classified[for_verification]  # svm_reshape

    print('Confusion matrix: \n %s' %
          confusion_matrix(verification_labels, predicted_labels))
    print('\n')

    print('Classificaion report: \n %s' %
          classification_report(verification_labels, predicted_labels, target_names=target_names))

    return confusion_matrix, classification_report



if __name__ == "__main__":

    raster_input_data = os.path.join(DATA_PATH, 'input', 'Sentinel2_20180630.tif')
    raster_output = os.path.join(DATA_PATH, 'output', 'SVM_urban_extraction.tif')
    raster = load_raster(raster_input_data)
    # select shapefiles that contains training data
    glob_path = glob.glob(os.path.join(DATA_PATH, 'training_data', '*'))
    shapefiles = [f for f in glob_path if f.endswith('.shp')]

    clf = tune('random')
    svm_classified = parallel_processing()
    urban_extraction = urban_extraction(raster_input_data)
    accuracy = model_accuracy()

    write_raster(raster, urban_extraction, raster_output)
    

