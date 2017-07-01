from __future__ import print_function

import os
import sys
sys.setrecursionlimit(10000)
import time
import json
import argparse
#from densenet169 import densenet169_model
import densenet


import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import scipy
from sklearn.metrics import fbeta_score
from sklearn.cross_validation import train_test_split

import keras.backend as K

from keras.optimizers import Adam
from keras.utils import np_utils


from keras.models import Model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape, core, Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


PLANET_KAGGLE_ROOT = os.path.abspath("../../input/")
PLANET_KAGGLE_TESTING_JPEG_TRAIN_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'testing-sets-for-coding/train-jpg-small')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
PLANET_KAGGLE_TESTING_JPEG_TEST_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'testing-sets-for-coding/test-jpg-small')
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_TESTING_JPEG_TRAIN_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)
assert os.path.exists(PLANET_KAGGLE_TESTING_JPEG_TEST_DIR)
test_submission_format_file = os.path.join(PLANET_KAGGLE_ROOT,'sample_submission_v2.csv')
assert os.path.isfile(test_submission_format_file)

df_train = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
df_test = pd.read_csv(test_submission_format_file)

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

NUM_CLASSES = len(labels)
THRESHHOLD = [0.2]*17
THRESHHOLD = np.array(THRESHHOLD)

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

X_train = []
X_test = []
y_train = []

print("Loading training set:\n")
for f, tags in tqdm(df_train.values[:99], miniters=10):
    img_path = PLANET_KAGGLE_TESTING_JPEG_TRAIN_DIR + '/{}.jpg'.format(f)
    img = cv2.imread(img_path)
    targets = np.zeros(NUM_CLASSES)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    X_train.append(img)
    y_train.append(targets)
    
X_train = np.array(X_train, np.float32)
y_train = np.array(y_train, int)

print('Training data shape: {}'  .format(X_train.shape))
print('Traing label shape: {}' .format(y_train.shape))

print("Loading test set:\n")
for f, tags in tqdm(df_test.values[:99], miniters=10):
    img_path = PLANET_KAGGLE_TESTING_JPEG_TEST_DIR + '/{}.jpg'.format(f)
    img = cv2.imread(img_path)
    X_test.append(img)

X_test = np.array(X_test, np.float32)
print('Test data shape: {}'  .format(X_test.shape))

###################
# Data processing #
###################

img_dim = X_train.shape[1:]

if K.image_dim_ordering() == "th":
    n_channels = X_train.shape[1]
else:
    n_channels = X_train.shape[-1]

if K.image_dim_ordering() == "th":
    for i in range(n_channels):
        mean_train = np.mean(X_train[:, i, :, :])
        std_train = np.std(X_train[:, i, :, :])
        X_train[:, i, :, :] = (X_train[:, i, :, :] - mean_train) / std_train
        
        mean_test = np.mean(X_test[:, i, :, :])
        std_test = np.std(X_test[:, i, :, :])
        X_test[:, i, :, :] = (X_test[:, i, :, :] - mean_test) / std_test
                    
elif K.image_dim_ordering() == "tf":
    for i in range(n_channels):
        mean_train = np.mean(X_train[:, :, :, i])
        std_train = np.std(X_train[:, :, :, i])
        X_train[:, :, :, i] = (X_train[:, :, :, i] - mean_train) / std_train
        
        mean_test = np.mean(X_test[:, :, :, i])
        std_test = np.std(X_test[:, :, :, i])
        X_test[:, :, :, i] = (X_test[:, :, :, i] - mean_test) / std_test
        
print('Splitting to training data set and validation set:')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
print('Splitted training data set shape: {}'  .format(X_train.shape))
print('Validation data set shape: {}' .format(X_val.shape))

#############
# Metrics #
############

def f2_beta(y_true, y_pred):
   return fbeta_score(y_true, y_pred, beta=2, average='samples')


def get_optimal_threshhold(y_true, y_pred, iterations = 100):
    best_threshhold = [0.2]*17    
    for t in range(NUM_CLASSES):
        best_fbeta = 0
        temp_threshhold = [0.2]*NUM_CLASSES
        for i in range(iterations):
            temp_value = i / float(iterations)
            temp_threshhold[t] = temp_value
            temp_fbeta = f2_beta(y_true, y_pred > temp_threshhold)
            if  temp_fbeta > best_fbeta:
                best_fbeta = temp_fbeta
                best_threshhold[t] = temp_value
    return best_threshhold


def f2_beta_keras(y_true, y_pred):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    TR_tf = tf.cast(tf.constant(THRESHHOLD),tf.float32)
    # y_pred_bin = K.round( tf.add( y_pred ,TR_tf) )  

    y_pred_bin = tf.cast(tf.greater(y_pred,TR_tf),tf.float32)


    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon())


###################
# Construct model #
###################

def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=img_dim))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
                  optimizer='adam',
                  metrics=['accuracy'])
                  

    return model


#model = densenet169_model(img_dim, num_classes=NUM_CLASSES)
#model = cnn_model()
# model.summary()
# model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
#                 optimizer='adam',
#                 metrics=['accuracy'])
# model.fit(X_train, y_train,
#           batch_size=32, epochs=1, shuffle=False,
#           validation_data=(X_val, y_val))
batch_size = 128
epochs = 30

depth = 40
nb_dense_block = 4
growth_rate = 12
nb_filter = 16
dropout_rate = 0.2 # 0.0 for data augmentation
weight_decay=1E-4

model = densenet.DenseNet(input_shape=img_dim, depth=depth, nb_dense_block=nb_dense_block, 
        growth_rate=growth_rate, nb_filter=nb_filter, nb_layers_per_block=-1,
        bottleneck=False, reduction=0.0, dropout_rate=dropout_rate, weight_decay=weight_decay,
        include_top=True, weights=None, input_tensor=None,
        classes=NUM_CLASSES, activation='softmax')
print("Model created")

model.summary()
optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',f2_beta_keras])
print("Finished compiling")
print("Building model...")

model.fit(X_train, y_train,
          batch_size=batch_size, epochs=epochs, shuffle=False,
          validation_data=(X_val, y_val))

y_pred_val = model.predict(X_val, batch_size=batch_size)
THRESHHOLD = get_optimal_threshhold(y_val, y_pred, iterations = 100)

y_pred = model.predict(X_test, batch_size=batch_size)
predictions = [' '.join(labels[y_pred > THRESHHOLD]) for y_pred_row in y_pred]

submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = preds
submission.to_csv('../results/submission_DenseNet_1.csv', index=False)






