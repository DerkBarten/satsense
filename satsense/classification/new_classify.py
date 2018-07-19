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
from sklearn.metrics import confusion_matrix

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
feature_vector_filenames = config['Features']['feature_vectors'].split(",")
feature_base_path = config['Features']['base_path']

# Mask settings
slum_mask_filenames = config['Masks']['slum_masks'].split(",")
building_mask_filenames = config['Masks']['building_masks'].split(",")
mask_base_path = config['Masks']['base_path']

LABELS = {
    'NON-SLUM': 0,
    'SLUM': 1
}

combined_images = train_images + [test_image]
print(combined_images)

satellite_images = []
for imagefile in combined_images:
    path = os.path.join(image_base_path, imagefile.strip())
    satellite_images.append(SatelliteImage.load_from_file(path, bands))

# Creating train set
X = None
y = None
g = None
for i, image in enumerate(satellite_images):
    slum_mask_path = os.path.join(mask_base_path, slum_mask_filenames[i].strip())
    building_mask_path = os.path.join(mask_base_path, building_mask_filenames[i].strip())
    slum_mask = Mask.load_from_file(slum_mask_path)\
                    .resample(tile_size, threshold)
    building_mask = Mask.load_from_file(building_mask_path)\
                        .resample(tile_size, threshold)
    feature_vector = np.load(os.path.join(feature_base_path,
                                          feature_vector_filenames[i].strip()))
   
    X_0, y_0 = Dataset(feature_vector).createXY(building_mask,
                                                in_label=LABELS['NON-SLUM'])
    X_1, y_1 = Dataset(feature_vector).createXY(slum_mask,
                                                in_label=LABELS['SLUM'])
    g_0 = np.full(y_0.shape, i)
    g_1 = np.full(y_1.shape, i)

    if X is None:
        X = np.concatenate((X_0, X_1), axis=0)
        # X = X_0
    else:
        X  = np.concatenate((X, X_0, X_1), axis=0)
        # X  = np.concatenate((X, X_0), axis=0)
    
    if y is None:
        y = np.concatenate((y_0, y_1), axis=0)
        # y = y_0
    else:
        y = np.concatenate((y, y_0, y_1), axis=0)
        # y = np.concatenate((y, y_0), axis=0)
    
    if g is None:
        g = np.concatenate((g_0, g_1), axis=0)
        # g = g_0
    else:
        g = np.concatenate((g, g_0, g_1), axis=0)
        # g = np.concatenate((g, g_0), axis=0)


print("unique y", np.unique(y, return_counts=True))



classifier = GradientBoostingClassifier()
cv = LeaveOneGroupOut()
print("start cross val predict")
y_pred = cross_val_predict(classifier, X, y, groups=g, cv=cv, n_jobs=-1)



slum_mask_path = os.path.join(mask_base_path, slum_mask_filenames[2].strip())
building_mask_path = os.path.join(mask_base_path, building_mask_filenames[2].strip())
slum_mask = Mask.load_from_file(slum_mask_path)\
                .resample(tile_size, threshold)
building_mask = Mask.load_from_file(building_mask_path)\
                    .resample(tile_size, threshold)

combi = Mask(slum_mask) | Mask(building_mask)

foo = y_pred[g == 2]
print(foo.shape)
print(combi.shape)
print(combi.mask[combi.mask == 1].shape)
print(np.unique(combi.mask, return_counts=True))
#combi[combi == 0] foo
print("--------")
print(matthews_corrcoef(y, y_pred))
print(confusion_matrix(y, y_pred))

print(y_pred.shape)
print(y_pred[g == 0].shape)
print(y_pred[g == 1].shape)
print(y_pred[g == 2].shape)

# plt.imshow(np.reshape(y_pred[g == 2], feature_vector.shape[:2]))
# plt.show()








