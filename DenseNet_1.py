from __future__ import print_function

import os
import sys
sys.setrecursionlimit(10000)
import time
import json
import argparse
import densenet
import tensorflow as tf
import math
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import scipy
from sklearn.metrics import fbeta_score
from sklearn.cross_validation import train_test_split

import keras.backend as K

from keras.optimizers import Adam
from keras.optimizers import SGD

from keras.utils import np_utils


from keras.models import Model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape, core, Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,TensorBoard,CSVLogger


PLANET_KAGGLE_ROOT = os.path.abspath("../../input/")
# PLANET_KAGGLE_TEST_JPEG_DIR  = os.path.join(PLANET_KAGGLE_ROOT, 'testing-sets-for-coding/test-jpg-small')
# PLANET_KAGGLE_TRAIN_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'testing-sets-for-coding/train-jpg-small')

PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
PLANET_KAGGLE_TRAIN_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg/')
PLANET_KAGGLE_TEST_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'test-jpg/')
test_submission_format_file = os.path.join(PLANET_KAGGLE_ROOT,'sample_submission_v2.csv')

assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)
assert os.path.isfile(test_submission_format_file)

assert os.path.exists(PLANET_KAGGLE_TRAIN_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_TEST_JPEG_DIR)

# assert os.path.exists(PLANET_KAGGLE_TESTING_JPEG_TRAIN_DIR)
# assert os.path.exists(PLANET_KAGGLE_TESTING_JPEG_TEST_DIR)


df_train = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)
df_test = pd.read_csv(test_submission_format_file)

flatten = lambda l: [item for sublist in l for item in sublist]
labels = np.array(list(set(flatten([l.split(' ') for l in df_train['tags'].values]))))

NUM_CLASSES = len(labels)
THRESHHOLD = [0.2]*17
THRESHHOLD = np.array(THRESHHOLD)

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

X_train = []
X_test = []
y_train = []

print("Loading training set:\n")
for f, tags in tqdm(df_train.values, miniters=100):
    img_path = PLANET_KAGGLE_TRAIN_JPEG_DIR + '/{}.jpg'.format(f)
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


###################
# Data processing #
###################

img_dim = X_train.shape[1:]

if K.image_dim_ordering() == "th":
    n_channels = X_train.shape[1]
elif K.image_dim_ordering() == "tf":
    n_channels = X_train.shape[-1]

if K.image_dim_ordering() == "th":
    for i in range(n_channels):
        mean_train = np.mean(X_train[:, i, :, :])
        std_train = np.std(X_train[:, i, :, :])
        X_train[:, i, :, :] = (X_train[:, i, :, :] - mean_train) / std_train
                            
elif K.image_dim_ordering() == "tf":
    for i in range(n_channels):
        mean_train = np.mean(X_train[:, :, :, i])
        std_train = np.std(X_train[:, :, :, i])
        X_train[:, :, :, i] = (X_train[:, :, :, i] - mean_train) / std_train
        
        
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
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
    #               optimizer='adam',
    #               metrics=['accuracy'])
                  

    return model


# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
# learning schedule callback
lrate = LearningRateScheduler(step_decay)

batch_size = 128
epochs = 50

learningrate = 0.1
decay = learningrate / epochs

depth = 40
nb_dense_block = 4
growth_rate = 48
nb_filter = 16
dropout_rate = 0.2 # 0.0 for data augmentation
weight_decay=1E-4

model = cnn_model()

# model = densenet.DenseNet(input_shape=img_dim, depth=depth, nb_dense_block=nb_dense_block, 
#         growth_rate=growth_rate, nb_filter=nb_filter, nb_layers_per_block=-1,
#         bottleneck=True, reduction=0.0, dropout_rate=dropout_rate, weight_decay=weight_decay,
#         include_top=True, weights=None, input_tensor=None,
#         classes=NUM_CLASSES, activation='softmax')
print("Model created")

model.summary()

# optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
optimizer = SGD(lr=learningrate, decay=0.0, momentum=0.9, nesterov=True)

model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',f2_beta_keras])
print("Finished compiling")
print("Building model...")

model_file_path = './model/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
check = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

tensorboard = TensorBoard(log_dir='./logs',write_graph=True, write_images=False)

log_filename = './logs/training.csv'
csv_logger = CSVLogger(log_filename,separator=',',append=False)
model.fit(X_train, y_train,
          batch_size=batch_size, epochs=epochs, shuffle=False,
          validation_data=(X_val, y_val),
          callbacks=[lrate,csv_logger,tensorboard])


del X_train
del y_train

print("Loading test set:\n")
for f, tags in tqdm(df_test.values, miniters=100):
    img_path = PLANET_KAGGLE_TEST_JPEG_DIR + '/{}.jpg'.format(f)
    img = cv2.imread(img_path)
    X_test.append(img)

X_test = np.array(X_test, np.float32)
print('Test data shape: {}'  .format(X_test.shape))

if K.image_dim_ordering() == "th":
    for i in range(n_channels):        
        mean_test = np.mean(X_test[:, i, :, :])
        std_test = np.std(X_test[:, i, :, :])
        X_test[:, i, :, :] = (X_test[:, i, :, :] - mean_test) / std_test
                    
elif K.image_dim_ordering() == "tf":
    for i in range(n_channels):
        
        mean_test = np.mean(X_test[:, :, :, i])
        std_test = np.std(X_test[:, :, :, i])
        X_test[:, :, :, i] = (X_test[:, :, :, i] - mean_test) / std_test





y_pred = model.predict(X_test, batch_size=batch_size)

predictions = [' '.join(labels[y_pred_row > 0.01]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('../../results/submission_CNN_1_THRESHHOLD_001.csv', index=False)


predictions = [' '.join(labels[y_pred_row > 0.05]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('../../results/submission_CNN_1_THRESHHOLD_005.csv', index=False)


predictions = [' '.join(labels[y_pred_row > 0.10]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('../../results/submission_CNN_1_THRESHHOLD_01.csv', index=False)

predictions = [' '.join(labels[y_pred_row > 0.20]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('../../results/submission_CNN_1_THRESHHOLD_02.csv', index=False)

y_pred_val = model.predict(X_val, batch_size=batch_size)
THRESHHOLD = get_optimal_threshhold(y_val, y_pred_val, iterations = 100)
THRESHHOLD = np.array(THRESHHOLD)
predictions = [' '.join(labels[y_pred_row > THRESHHOLD]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('../../results/submission_CNN_THRESHOLD_OPTIMAL.csv', index=False)








