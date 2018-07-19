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
image_base_path = config['Image']['base_path']
train_images = config['Image']['train_images'].split(",")
test_image = config['Image']['test_image']
bands = eval(config['Image']['bands'])

# Feature settings
tile_size = eval(config['Features']['tile_size'])
threshold = eval(config['Features']['threshold'])
window_sizes = eval(config['Features']['window_sizes'])
train_feature_vector_filenames = config['Features']['train_feature_vectors'].split(",")
test_feature_vector_filename = config['Features']['test_feature_vector']
feature_base_path = config['Features']['base_path']

# Mask settings
train_slum_mask_filenames = config['Masks']['train_slum_masks'].split(",")
test_slum_mask_filename = config['Masks']['test_slum_mask']
train_building_mask_filenames = config['Masks']['train_building_masks'].split(",")
test_building_mask_filename = config['Masks']['test_building_mask']
train_vegetation_mask_filenames = config['Masks']['train_vegetation_masks'].split(",")
test_vegetation_mask_filename = config['Masks']['test_vegetation_mask']
mask_base_path = config['Masks']['base_path']

LABELS = {
    'BUILDING': 1,
    'SLUM': 2,
    'VEGETATION': 3
}

train_satellite_images = []
for imagefile in train_images:
    path = os.path.join(image_base_path, imagefile.strip())
    print(path)
    train_satellite_images.append(SatelliteImage.load_from_file(path, bands))

test_satellite_image = SatelliteImage.load_from_file(os.path.join(
                            image_base_path.strip(), test_image), bands)

print("Creating train set...")
X_train = None
y_train = None
for i, satellite_image in enumerate(train_satellite_images):
    vegetation_mask_path = os.path.join(mask_base_path, train_vegetation_mask_filenames[i].strip())
    slum_mask_path = os.path.join(mask_base_path, train_slum_mask_filenames[i].strip())
    building_mask_path = os.path.join(mask_base_path, train_building_mask_filenames[i].strip())
    vegetation_mask = Mask.load_from_file(vegetation_mask_path)\
                    .resample(tile_size, threshold)
    slum_mask = Mask.load_from_file(slum_mask_path)\
                    .resample(tile_size, threshold)
    building_mask = Mask.load_from_file(building_mask_path)\
                        .resample(tile_size, threshold)
    feature_vector = np.load(os.path.join(feature_base_path,
                                          train_feature_vector_filenames[i].strip()))
    # feature_vector = slum_mask[:, :, np.newaxis]

    X_0, y_0 = Dataset(feature_vector).createXY(building_mask,
                                                in_label=LABELS['BUILDING'])
    X_1, y_1 = Dataset(feature_vector).createXY(slum_mask,
                                                in_label=LABELS['SLUM'])
    X_2, y_2 = Dataset(feature_vector).createXY(vegetation_mask,
                                                in_label=LABELS['VEGETATION'])

    if X_train is None:
        X_train = np.concatenate((X_0, X_1, X_2), axis=0)
        # X_train = X_1
    else:
        X_train  = np.concatenate((X_train, X_0, X_1, X_2), axis=0)
        # X_train  = np.concatenate((X_train, X_1), axis=0)
    
    if y_train is None:
        y_train = np.concatenate((y_0, y_1, y_2), axis=0)
        # y_train = y_1
    else:
        y_train = np.concatenate((y_train, y_0, y_1, y_2), axis=0)
        # y_train = np.concatenate((y_train, y_1), axis=0)


print("before SMOTE")
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

#X_train, y_train = SMOTE().fit_sample(X_train, y_train)
print("after SMOTE")
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))



print("Creating test set...")
feature_vector = np.load(os.path.join(feature_base_path,
                                      test_feature_vector_filename.strip()))
slum_mask_path = os.path.join(mask_base_path, test_slum_mask_filename.strip())
building_mask_path = os.path.join(mask_base_path, test_building_mask_filename.strip())
vegetation_mask_path = os.path.join(mask_base_path, test_vegetation_mask_filename.strip())
slum_mask = Mask.load_from_file(slum_mask_path)\
                 .resample(tile_size, threshold)
building_mask = Mask.load_from_file(building_mask_path)\
                 .resample(tile_size, threshold)
vegetation_mask = Mask.load_from_file(vegetation_mask_path)\
                 .resample(tile_size, threshold)


ones_mask = Mask(np.ones(feature_vector.shape[:2]))
building_mask = ones_mask - (Mask(slum_mask) | Mask(vegetation_mask))
building_mask = building_mask.mask

y_test = np.zeros(feature_vector.shape[:2])
print(y_test.shape)
print(slum_mask.shape)
print(building_mask.shape)
print(vegetation_mask.shape)
plt.imshow(slum_mask)
plt.show()
plt.imshow(building_mask)
plt.show()
plt.imshow(vegetation_mask)
plt.show()
y_test[slum_mask == 1] =  LABELS['SLUM']
y_test[building_mask == 1] = LABELS['BUILDING']                                                       
y_test[vegetation_mask == 1] =  LABELS['VEGETATION']

print(np.unique(y_test, return_counts=True))

nrows = feature_vector.shape[0] * feature_vector.shape[1]
nfeatures = feature_vector.shape[2]

X_test = np.reshape(feature_vector, (nrows, nfeatures))
y_test = np.reshape(y_test, (nrows, ))


print(X_test.shape)
print(y_test.shape)

classifier = GradientBoostingClassifier()
print("fitting...")

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
y_pred[y_pred == LABELS['VEGETATION']] = LABELS['BUILDING']
y_test[y_test == LABELS['VEGETATION']] = LABELS['BUILDING']

print(matthews_corrcoef(y_test, y_pred))
result_mask = Mask(np.reshape(y_pred, feature_vector.shape[:2]))
result_mask.overlay(test_satellite_image.rgb)
plt.show()










