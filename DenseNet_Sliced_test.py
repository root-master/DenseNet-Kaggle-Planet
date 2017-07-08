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
from sklearn.model_selection import train_test_split

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

print('Splitting to training data set and validation set:')
df_train_split, df_val = train_test_split(df_train, test_size=0.1)

print('Splitted training data set size: {}'  .format(df_train_split.shape[0]))
print('Validation data set size: {}' .format(df_val.shape[0]))

print('Slicing training data spilt set into set of chuncks:')
chunk_size = 4096
chunks = df_train_split.shape[0] // chunk_size
train_slices = []
for idx in range(chunks):
    train_slices.append(slice(idx*chunk_size,(idx+1)*chunk_size))
train_slices.append(slice((idx+1)*chunk_size,None))
# train_slices = np.array_split(np.arange(df_train_split.shape[0]) , chunks+1)

print('Slicing test set into set of chuncks:')
chunk_size = 4096
chunks = df_test.shape[0] // chunk_size
test_slices = []
for idx in range(chunks):
    test_slices.append(slice(idx*chunk_size,(idx+1)*chunk_size))
test_slices.append(slice((idx+1)*chunk_size,None))
# test_slices = np.array_split(np.arange(df_test.shape[0]) , chunks+1)

flatten = lambda l: [item for sublist in l for item in sublist]
labels = np.array(list(set(flatten([l.split(' ') for l in df_train['tags'].values]))))

NUM_CLASSES = len(labels)
THRESHHOLD = [0.2]*17
THRESHHOLD = np.array(THRESHHOLD)

label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}

if K.image_dim_ordering() == "th": 
# if you want to use theano as backend, images should be reshaped.
# I haven't rehspaed in this script because I am using tensorflow
    n_channels = 3
    img_dim = (3,256,256)
elif K.image_dim_ordering() == "tf":
    n_channels = 3
    img_dim = (256,256,3)

def load_train_data_slice(data_slice):
    X_train = []
    y_train = []
    for f, tags in tqdm(df_train_split.values[data_slice], miniters=100):
        img_path = PLANET_KAGGLE_TRAIN_JPEG_DIR + '/{}.jpg'.format(f)
        img = cv2.imread(img_path)
        targets = np.zeros(NUM_CLASSES)
        for t in tags.split(' '):
            targets[label_map[t]] = 1 
        X_train.append(img)
        y_train.append(targets)
    
    X_train = np.array(X_train, np.float32)
    y_train = np.array(y_train, int)

    return X_train, y_train

def load_test_data_slice(data_slice):
    X_test = []
    for f, tags in tqdm(df_test.values[data_slice], miniters=100):
        img_path = PLANET_KAGGLE_TEST_JPEG_DIR + '/{}.jpg'.format(f)
        img = cv2.imread(img_path)
        X_test.append(img)
    
    X_test = np.array(X_train, np.float32)

    return X_test

def load_validation_data():
    X_val = []
    y_val = []
    print('Loading Validation set:')
    for f, tags in tqdm(df_val.values, miniters=100):
        img_path = PLANET_KAGGLE_TRAIN_JPEG_DIR + '/{}.jpg'.format(f)
        img = cv2.imread(img_path)
        targets = np.zeros(NUM_CLASSES)
        for t in tags.split(' '):
            targets[label_map[t]] = 1 
        X_val.append(img)
        y_val.append(targets)
    
    X_val = np.array(X_val, np.float32)
    y_val = np.array(y_val, int)

    return X_val, y_val

###################
# Data processing #
###################
def preprocess(X_train):
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
    return X_train

X_val, y_val = load_validation_data()
X_val = preprocess(X_val)


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

batch_size = 16
epochs = 1

learning_rate = 0.01
decay = learning_rate / epochs

depth = 24
nb_dense_block = 4
growth_rate = 12
nb_filter = 16
dropout_rate = 0.2 # 0.0 for data augmentation
weight_decay=1E-4

# model = cnn_model()

model = densenet.DenseNet(input_shape=img_dim, depth=depth, nb_dense_block=nb_dense_block, 
        growth_rate=growth_rate, nb_filter=nb_filter, nb_layers_per_block=-1,
        bottleneck=True, reduction=0.0, dropout_rate=dropout_rate, weight_decay=weight_decay,
        include_top=True, weights=None, input_tensor=None,
        classes=NUM_CLASSES, activation='softmax')
print("Model created")

model.summary()

# optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
# optimizer = SGD(lr=learning_rate, decay=0.0, momentum=0.9, nesterov=True)
optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)


model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',f2_beta_keras])
print("Finished compiling")
print("Building model...")

model_file_path = './model/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
check = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

tensorboard = TensorBoard(log_dir='./logs',write_graph=True, write_images=False)

log_filename = './logs/training.csv'
csv_logger = CSVLogger(log_filename,separator=',',append=False)
# model.fit(X_train, y_train,
#           batch_size=batch_size, epochs=epochs, shuffle=False,
#           validation_data=(X_val, y_val),
#           callbacks=[lrate,csv_logger,tensorboard])


####################
# Network training #
####################

print("Training")

list_train_loss = []
list_test_loss = []
list_learning_rate = []

for e in range(epochs):

    if e == int(0.5 * epochs):
        K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

    if e == int(0.75 * epochs):
        K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))
    
    l_train_loss = []   
    split_size = batch_size
    for train_slice in train_slices:
        X_train, y_train = load_train_data_slice(train_slice) 
        X_train = preprocess(X_train)

        num_splits = X_train.shape[0] / split_size
        arr_splits = np.array_split(np.arange(X_train.shape[0]), num_splits)

        start = time.time()

        for batch_idx in arr_splits:
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
            train_logloss, train_acc = model.train_on_batch(X_batch, y_batch)
            l_train_loss.append([train_logloss, train_acc])
            list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())

    val_loss, val_acc = model.evaluate(X_val,
                                        y_val,
                                        verbose=1,
                                        batch_size=batch_size)
        
    list_test_loss.append([val_loss, val_acc])
    list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
        # to convert numpy array to json serializable
    print('Epoch %s/%s, Time: %s' % (e + 1, epochs, time.time() - start))
    model.save('./model/last-epoch-model.h5')

    d_log = {}
    d_log["batch_size"] = batch_size
    d_log["nb_epoch"] = epochs
    d_log["optimizer"] = optimizer.get_config()
    d_log["train_loss"] = list_train_loss
    d_log["test_loss"] = list_test_loss
    d_log["learning_rate"] = list_learning_rate

    json_file = os.path.join('./logs/experiment_Planet_Densenet.json')
    with open(json_file, 'w') as fp:
        json.dump(d_log, fp, indent=4, sort_keys=True)




# for e in range(epochs):
#     print("epoch %d" % e)
#     for train_slice in train_slices[0]:
#         X_train, y_train = load_train_data_slice(train_slice) 
#         X_train = preprocess(X_train)
#         model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,
#                     callbacks=[lrate,csv_logger,tensorboard])
#     val_loss = model.evaluate(X_val, y_val, batch_size=batch_size, verbose=1)

# model.save('my_model.h5')

y_pred = np.zeros((df_test.values.shape[0],NUM_CLASSES))
for test_slice in test_slices:
    X_test = load_test_data_slice(test_slice)
    X_test = preprocess(X_test)
    y_pred[test_slice,:] = model.predict(X_test, batch_size=batch_size)
    
# print("Loading test set:\n")
# for f, tags in tqdm(df_test.values, miniters=100):
#     img_path = PLANET_KAGGLE_TEST_JPEG_DIR + '/{}.jpg'.format(f)
#     img = cv2.imread(img_path)
#     X_test.append(img)

# X_test = np.array(X_test, np.float32)
# print('Test data shape: {}'  .format(X_test.shape))

# if K.image_dim_ordering() == "th":
#     for i in range(n_channels):        
#         mean_test = np.mean(X_test[:, i, :, :])
#         std_test = np.std(X_test[:, i, :, :])
#         X_test[:, i, :, :] = (X_test[:, i, :, :] - mean_test) / std_test
                    
# elif K.image_dim_ordering() == "tf":
#     for i in range(n_channels):
        
#         mean_test = np.mean(X_test[:, :, :, i])
#         std_test = np.std(X_test[:, :, :, i])
#         X_test[:, :, :, i] = (X_test[:, :, :, i] - mean_test) / std_test
# y_pred = model.predict(X_test, batch_size=batch_size)

predictions = [' '.join(labels[y_pred_row > 0.02]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results/submission_CNN_1_THRESHHOLD_001.csv', index=False)


predictions = [' '.join(labels[y_pred_row > 0.05]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results/submission_CNN_1_THRESHHOLD_005.csv', index=False)


predictions = [' '.join(labels[y_pred_row > 0.10]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results/submission_CNN_1_THRESHHOLD_01.csv', index=False)

predictions = [' '.join(labels[y_pred_row > 0.20]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results/submission_CNN_1_THRESHHOLD_02.csv', index=False)

y_pred_val = model.predict(X_val, batch_size=batch_size)
THRESHHOLD = get_optimal_threshhold(y_val, y_pred_val, iterations = 100)
THRESHHOLD = np.array(THRESHHOLD)
predictions = [' '.join(labels[y_pred_row > THRESHHOLD]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results/submission_CNN_THRESHOLD_OPTIMAL.csv', index=False)








