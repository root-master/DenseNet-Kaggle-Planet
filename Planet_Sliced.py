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
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D,ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Reshape, core, Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,TensorBoard,CSVLogger

from VGG16 import VGG16


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

print('Splitting to training data set and validation set')
df_train_split, df_val = train_test_split(df_train, test_size=0.1)

print('Splitted training data set size: {}'  .format(df_train_split.shape[0]))
print('Validation data set size: {}' .format(df_val.shape[0]))

print('Slicing training data spilt set into set of chuncks')
chunk_size = 4096
chunks = df_train_split.shape[0] // chunk_size
train_slices = []
for idx in range(chunks):
    train_slices.append(slice(idx*chunk_size,(idx+1)*chunk_size))
train_slices.append(slice((idx+1)*chunk_size,None))
# train_slices = np.array_split(np.arange(df_train_split.shape[0]) , chunks+1)

print('Slicing all training data set into set of chuncks')
chunk_size = 4096
chunks = df_train.shape[0] // chunk_size
all_train_slices = []
for idx in range(chunks):
    all_train_slices.append(slice(idx*chunk_size,(idx+1)*chunk_size))
all_train_slices.append(slice((idx+1)*chunk_size,None))

print('Slicing test set into set of chuncks')
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

def load_test_data_slice(test_slice):
    X_test = []
    for f, tags in tqdm(df_test.values[test_slice], miniters=100):
        img_path = PLANET_KAGGLE_TEST_JPEG_DIR + '/{}.jpg'.format(f)
        img = cv2.imread(img_path)
        X_test.append(img)    
    X_test = np.array(X_test, np.float32)
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

##########################
# Loading Validation set 
# to memory 
##########################
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
epochs = 20

learning_rate = 0.001
decay = learning_rate / epochs

#########################################
############## DENSENET #################
#########################################
# depth = 100
# nb_dense_block = 4
# growth_rate = 12
# nb_filter = 16
# dropout_rate = 0.2 # 0.0 for data augmentation
# weight_decay=1E-4
# model = densenet.DenseNet(input_shape=img_dim, depth=depth, nb_dense_block=nb_dense_block, 
#         growth_rate=growth_rate, nb_filter=nb_filter, nb_layers_per_block=-1,
#         bottleneck=True, reduction=0.0, dropout_rate=dropout_rate, weight_decay=weight_decay,
#         include_top=True, weights=None, input_tensor=None,
#         classes=NUM_CLASSES, activation='softmax')


#######################################
########## LOAD MODEL #################
#######################################
model = VGG16(classes=NUM_CLASSES)
# model = cnn_model()
#model = VGG_16()
print("Model created")
model.summary()

#######################################
########## OPTIMIZER ##################
#######################################
# optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
optimizer = SGD(lr=learning_rate, decay=0.0, momentum=0.9, nesterov=True)
# optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy',f2_beta_keras])
print("Finished compiling")
print("Building model...")

######################################
####### LOG, SAVE AND MONITOR ########
######################################
model_file_path = './model/weights.{epoch:02d}-{val_loss:.2f}.hdf5'
check = ModelCheckpoint(model_file_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
tensorboard = TensorBoard(log_dir='./logs',write_graph=True, write_images=False)
log_filename = './logs/training.csv'
csv_logger = CSVLogger(log_filename,separator=',',append=False)
# model.fit(X_train, y_train,
#           batch_size=batch_size, epochs=epochs, shuffle=False,
#           validation_data=(X_val, y_val),
#           callbacks=[lrate,csv_logger,tensorboard])





##################################
###### DATA AUGMENTATION #########
##################################
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.,
    zoom_range=0.2,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=None,
    preprocessing_function=None,
    data_format=K.image_data_format())

####################
# Network training #
####################

print("Training")

list_train_loss = []
list_val_loss = []
list_learning_rate = []
val_loss = 1000
val_loss_min = 1000
val_loss_last = 1000
wait = 0 # using for early stopping

for e in tqdm(range(epochs),miniters=1,desc='Epochs'):
    start = time.time()
    if e == int(0.5 * epochs):
        K.set_value(model.optimizer.lr, np.float32(learning_rate / 10.))

    if e == int(0.75 * epochs):
        K.set_value(model.optimizer.lr, np.float32(learning_rate / 100.))
    
    l_train_loss = []   
    split_size = batch_size

    ########### LOOP ON TRAINING DATA SLICES ###########
    for train_slice in tqdm(train_slices,miniters=1,desc='Train Samples'):
        X_train, y_train = load_train_data_slice(train_slice) 
        X_train = preprocess(X_train)
        datagen.fit(X_train)
        batches = 0
        ############ LOOP ON BATCHES ###########
        for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):           
            train_logloss, train_acc, f2_score = model.train_on_batch(X_batch, y_batch)
            l_train_loss.append([train_logloss, train_acc, f2_score])
            list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
            batches += 1
            if batches >= len(X_train) / batch_size:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break
        
        # num_splits = X_train.shape[0] / split_size
        # arr_splits = np.array_split(np.arange(X_train.shape[0]), num_splits)
        # start = time.time()
        # for batch_idx in tqdm(arr_splits,miniters=1,desc='Batch'):
        #     X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
        #     train_logloss, train_acc,f2_score = model.train_on_batch(X_batch, y_batch)
        #     l_train_loss.append([train_logloss, train_acc, f2_score])
        #     list_train_loss.append(np.mean(np.array(l_train_loss), 0).tolist())
            

    batches = 0
    datagen.fit(X_val)
    generator = datagen.flow(X_val, y_val, batch_size=batch_size)

    val_loss_last = val_loss
    val_loss, val_acc,val_f2_score = \
            model.evaluate_generator(generator,steps_per_epoch=len(X_val) / batch_size)
    
    # save the best model based on validation loss
    if val_loss < val_loss_min:
        val_loss_min = val_loss
        wait = 0 # restart wait
        model.save('best-epoch-model-min-val-loss.h5')

    l_val_loss.append([val_loss, val_acc, val_f2_score])
    
    list_val_loss.append(np.mean(np.array(l_val_loss), 0).tolist())
    ############ LOOP ON BATCHES of Validation for DATA AUGMENTATION ###########
    # l_val_loss = []
    # for X_batch, y_batch in datagen.flow(X_val, y_val, batch_size=batch_size):           
    #     val_loss, val_acc,val_f2_score = model.test_on_batch(X_batch, y_batch)
    #     l_val_loss.append([val_loss, val_acc, val_f2_score])
    #     list_val_loss.append(np.mean(np.array(l_val_loss), 0).tolist())
    
    #     batches += 1
    #     if batches >= len(X_train) / batch_size:
    #         # we need to break the loop by hand because
    #         # the generator loops indefinitely
    #         break

    # val_loss, val_acc,val_f2_score = model.evaluate(X_val,
    #                                     y_val,
    #                                     verbose=1,
    #                                     batch_size=batch_size)        
    # list_test_loss.append([val_loss, val_acc,val_f2_score])
    # list_learning_rate.append(float(K.get_value(model.optimizer.lr)))
        # to convert numpy array to json serializable
    print('Epoch %s/%s, Time: %s' % (e + 1, epochs, time.time() - start))

    d_log = {}
    d_log["batch_size"] = batch_size
    d_log["nb_epoch"] = epochs
    d_log["optimizer"] = optimizer.get_config()
    d_log["train_loss"] = list_train_loss
    d_log["test_loss"] = list_val_loss
    d_log["learning_rate"] = list_learning_rate

    json_file = os.path.join('./logs/experiment_Planet_Densenet.json')
    with open(json_file, 'w') as fp:
        json.dump(d_log, fp, indent=4, sort_keys=True)

    model.save('last-epoch-model-val.h5')
    # early stopping
    if val_loss > val_loss_last:
        wait += 1

    if wait == 2:
        break
        
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
y_pred_not_aug = np.zeros((df_test.values.shape[0],NUM_CLASSES))

for test_slice in tqdm(test_slices,miniters=1,desc='Prediction'):
    X_test = load_test_data_slice(test_slice)
    X_test = preprocess(X_test)
    y_pred_not_aug[test_slice,:] = model.predict(X_test, batch_size=batch_size,verbose=1)

    datagen.fit(X_test)
    generator = datagen.flow(X_test,batch_size=batch_size)
    y_pred[test_slice,:] = \
        model.predict_generator(generator, steps=len(X_test)/batch_size, max_queue_size=10, workers=1, use_multiprocessing=True, verbose=1)



##################### PREDICTIONS NOT AUGMENTED ################### 
########### Split data 90% - 10% Validation #######################
predictions = [' '.join(labels[y_pred_row >= 0.1]) for y_pred_row in y_pred_not_aug]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results-VGG/submission_VGG16_THRESHHOLD_01_not_aug.csv', index=False)


predictions = [' '.join(labels[y_pred_row >= 0.2]) for y_pred_row in y_pred_not_aug]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results-VGG/submission_VGG16_THRESHHOLD_02_not_aug.csv', index=False)

predictions = [' '.join(labels[y_pred_row >= 0.5]) for y_pred_row in y_pred_not_aug]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results-VGG/submission_VGG16_THRESHHOLD_05_not_aug.csv', index=False)


y_pred_val = model.predict(X_val, batch_size=batch_size)
THRESHHOLD = get_optimal_threshhold(y_val, y_pred_val, iterations = 100)
THRESHHOLD = np.array(THRESHHOLD)
predictions = [' '.join(labels[y_pred_row >= THRESHHOLD]) for y_pred_row in y_pred_not_aug]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results-VGG/submission_VGG16_THRESHOLD_OPTIMAL_not_aug.csv', index=False)






##################### PREDICTIONS AUGMENTED #######################
########### Split data 90% - 10% Validation #######################
predictions = [' '.join(labels[y_pred_row >= 0.1]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results-VGG/submission_VGG16_THRESHHOLD_01.csv', index=False)


predictions = [' '.join(labels[y_pred_row >= 0.2]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results-VGG/submission_VGG16_THRESHHOLD_02.csv', index=False)

predictions = [' '.join(labels[y_pred_row >= 0.5]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results-VGG/submission_VGG16_THRESHHOLD_05.csv', index=False)


y_pred_val = model.predict(X_val, batch_size=batch_size)
THRESHHOLD = get_optimal_threshhold(y_val, y_pred_val, iterations = 100)
THRESHHOLD = np.array(THRESHHOLD)
predictions = [' '.join(labels[y_pred_row >= THRESHHOLD]) for y_pred_row in y_pred]
submission = pd.DataFrame()
submission['image_name'] = df_test.image_name.values
submission['tags'] = predictions
submission.to_csv('./results-VGG/submission_VGG16_THRESHOLD_OPTIMAL.csv', index=False)