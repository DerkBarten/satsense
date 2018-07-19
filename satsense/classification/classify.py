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
mask_base_path = config['Masks']['base_path']

LABELS = {
    'NON-SLUM': 0,
    'SLUM': 1
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
    slum_mask_path = os.path.join(mask_base_path, train_slum_mask_filenames[i].strip())
    building_mask_path = os.path.join(mask_base_path, train_building_mask_filenames[i].strip())
    slum_mask = Mask.load_from_file(slum_mask_path)\
                    .resample(tile_size, threshold)
    building_mask = Mask.load_from_file(building_mask_path)\
                        .resample(tile_size, threshold)
    feature_vector = np.load(os.path.join(feature_base_path,
                                          train_feature_vector_filenames[i].strip()))
    # feature_vector = slum_mask[:, :, np.newaxis]

    X_0, y_0 = Dataset(feature_vector).createXY(building_mask,
                                                in_label=LABELS['NON-SLUM'])
    X_1, y_1 = Dataset(feature_vector).createXY(slum_mask, remove_out=False,
                                                in_label=LABELS['SLUM'],
                                                out_label=LABELS['NON-SLUM'])

    if X_train is None:
        # X_train = np.concatenate((X_0, X_1), axis=0)
        X_train = X_1
    else:
        # X_train  = np.concatenate((X_train, X_0, X_1), axis=0)
        X_train  = np.concatenate((X_train, X_1), axis=0)
    
    if y_train is None:
        # y_train = np.concatenate((y_0, y_1), axis=0)
        y_train = y_1
    else:
        # y_train = np.concatenate((y_train, y_0, y_1), axis=0)
        y_train = np.concatenate((y_train, y_1), axis=0)


print("before SMOTE")
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))

X_train, y_train = SMOTE().fit_sample(X_train, y_train)
print("after SMOTE")
print(X_train.shape)
print(y_train.shape)
print(np.unique(y_train, return_counts=True))



print("Creating test set...")
feature_vector = np.load(os.path.join(feature_base_path,
                                      test_feature_vector_filename.strip()))
slum_mask_path = os.path.join(mask_base_path, test_slum_mask_filename.strip())
building_mask_path = os.path.join(mask_base_path, test_building_mask_filename.strip())
slum_mask = Mask.load_from_file(slum_mask_path)\
                 .resample(tile_size, threshold)
building_mask = Mask.load_from_file(building_mask_path)\
                 .resample(tile_size, threshold)

# Mask(slum_mask).overlay(test_satellite_image.rgb)
# plt.show()
#feature_vector = slum_mask[:, :, np.newaxis]


X_test, y_test = Dataset(feature_vector).createXY(slum_mask, remove_out=False,
                                                  in_label=LABELS['SLUM'],
                                                  out_label=LABELS['NON-SLUM'])

print(X_test.shape)
print(y_test.shape)

classifier = GradientBoostingClassifier()
print("fitting...")

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(matthews_corrcoef(y_test, y_pred))
result_mask = Mask(np.reshape(y_pred, feature_vector.shape[:2]))
result_mask.overlay(test_satellite_image.rgb)

plt.show()


# print(X_train.shape)
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)
# print(y_test)
#y_pred = np.reshape(y_test, feature_vector.shape[:2])








