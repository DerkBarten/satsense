from satsense.image import SatelliteImage
from satsense.generators import CellGenerator
from satsense.features import FeatureSet, HistogramOfGradients
from satsense import extract_features
from satsense import WORLDVIEW3
from satsense.masks import Mask
from satsense.classification import Dataset
import os
import numpy as np
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef

import configparser
config = configparser.ConfigParser()
config.read("classify.ini")

# Image settings
IMAGE_FOLDER = config['Image']['folder']
BANDS = eval(config['Image']['bands'])
TRAIN_IMAGES = [n.strip() for n in config['Image']['train_images'].split(",")]
TEST_IMAGE = config['Image']['test_image'].strip()

# Feature settings
TILE_SIZE = eval(config['Features']['tile_size'])
THRESHOLD = eval(config['Features']['threshold'])
FEATURE_FOLDER = config['Features']['folder']
MASK_FOLDER = config['Masks']['folder']

LABELS = {
    'BUILDING': 1,
    'SLUM': 2,
    'VEGETATION': 3
}

def load_masks(image_name):
    path = os.path.join(MASK_FOLDER, "vegetation", image_name + ".npy")
    vegetation_mask = Mask.load_from_file(path)
    path = os.path.join(MASK_FOLDER, "slum", image_name + ".npy")
    slum_mask = Mask.load_from_file(path)
    
    slum_mask = slum_mask.resample(TILE_SIZE, THRESHOLD)
    vegetation_mask = vegetation_mask.resample(TILE_SIZE, THRESHOLD)

    building_mask = (Mask(np.ones(slum_mask.shape)) - (Mask(vegetation_mask) | Mask(slum_mask))).mask

    return slum_mask, vegetation_mask, building_mask

def load_feature_vector(image_name):
    return np.load(os.path.join(FEATURE_FOLDER, image_name + ".npy"))


def create_training_set():
    X_train = None
    y_train = None
    for imagefile in TRAIN_IMAGES:
        image_name = os.path.splitext(imagefile)[0]

        slum_mask, vegetation_mask, building_mask = load_masks(image_name)
        feature_vector = load_feature_vector(image_name)
        
        X_0, y_0 = Dataset(feature_vector).createXY(building_mask,
                                                    in_label=LABELS['BUILDING'])
        X_1, y_1 = Dataset(feature_vector).createXY(slum_mask,
                                                    in_label=LABELS['SLUM'])
        X_2, y_2 = Dataset(feature_vector).createXY(vegetation_mask,
                                                    in_label=LABELS['VEGETATION'])

        if X_train is None:
            X_train = np.concatenate((X_0, X_1, X_2), axis=0)
        else:
            X_train  = np.concatenate((X_train, X_0, X_1, X_2), axis=0)
        
        if y_train is None:
            y_train = np.concatenate((y_0, y_1, y_2), axis=0)
        else:
            y_train = np.concatenate((y_train, y_0, y_1, y_2), axis=0)
    
    return X_train, y_train

def create_test_set():
    print(np.unique(y_train, return_counts=True))

    image_name = os.path.splitext(TEST_IMAGE)[0]
    slum_mask, vegetation_mask, building_mask = load_masks(image_name)
    feature_vector = load_feature_vector(image_name)

    y_test = np.zeros(feature_vector.shape[:2])
    y_test[slum_mask == 1] =  LABELS['SLUM']
    y_test[building_mask == 1] = LABELS['BUILDING']                                                       
    y_test[vegetation_mask == 1] =  LABELS['VEGETATION']

    nrows = feature_vector.shape[0] * feature_vector.shape[1]
    nfeatures = feature_vector.shape[2]

    X_test = np.reshape(feature_vector, (nrows, nfeatures))
    y_test = np.reshape(y_test, (nrows, ))

    return X_test, y_test

if __name__ == "__main__":
    print("Creating training set...")
    X_train, y_train = create_training_set()
    print("Creating test set...")
    X_test, y_test = create_test_set()

    classifier = GradientBoostingClassifier()
    print("fitting...")

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    # Label the vegetation as buildings to create more accurate representation of the performance
    y_pred[y_pred == LABELS['VEGETATION']] = LABELS['BUILDING']
    y_test[y_test == LABELS['VEGETATION']] = LABELS['BUILDING']

    print(matthews_corrcoef(y_test, y_pred))
    result_mask = Mask(np.reshape(y_pred, feature_vector.shape[:2]))
    test_image = SatelliteImage.load_from_file(TEST_IMAGE, BANDS)
    result_mask.overlay(test_image.rgb)
    plt.show()










