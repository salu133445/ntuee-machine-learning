# For NTU 2016 Fall Machine Learning Class
# Homework 3: Semi-supervised photo classifier using CIFAR-10 database
# Program 3: semi-supervised photo classifier by CNN using auto-encoder and self-training
# Author: Herman Dong
# Tested under Python 2.7.9 with Keras 1.1.1 in Linux environment

import argparse
import numpy as np
import pickle
from keras.models import Sequential, model_from_json, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Input
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2, l1l2
from keras.backend import set_image_dim_ordering
set_image_dim_ordering('th') #using (samples, channels, width, height)

""" parser """

parser = argparse.ArgumentParser()
parser.add_argument( "dataDirectory", help='directory that contains training/test data' )
parser.add_argument( "outputModel", help='postfix of the output model' )
args = parser.parse_args()
postfix = args.outputModel

""" preprocessing stage"""

# define parameter
nb_classes = 10
train_size_labelled = 5000
train_size_unlabelled = 45000
test_size = 10000

# read train data
all_labelled = pickle.load( open( args.dataDirectory + 'all_label.p', 'rb' ) )
#all_unlabelled = pickle.load( open( args.dataDirectory + 'all_unlabel.p', 'rb' ) )
#all_test = pickle.load( open( args.dataDirectory + 'test.p', 'rb' ) )

# pre-process training data
x_train = np.asarray( all_labelled ).reshape( train_size_labelled, 3, 32, 32 )
#x_train_unlabelled = np.asarray( all_unlabelled ).reshape( train_size_unlabelled, 3, 32, 32 )
#x_test = np.asarray(all_test['data']).reshape( test_size, 3, 32, 32 )

#all_size = train_size_labelled + train_size_unlabelled + test_size
#x_all = np.concatenate(( x_train, x_train_unlabelled, x_test )) # combine 
#x_all = x_all.astype('float32') / 255.  # normalize training data's RGB value into [0,1]
x_train = x_train.astype('float32') / 255.

# create model 
input_img = Input(shape=(3,32,32))

x = Convolution2D(64, 3, 3, activation='sigmoid', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(64, 3, 3, activation='sigmoid', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

x = Convolution2D(64, 3, 3, activation='sigmoid', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(64, 3, 3, activation='sigmoid',border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

encoder = Model(input_img, encoded)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
autoencoder.summary()

""" training stage """
earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
weight_saved_path = 'weights_' + postfix + '.hdf5'
checkpoint = ModelCheckpoint(weight_saved_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
fitting = autoencoder.fit(x_train, x_train, nb_epoch=100, batch_size=128, validation_split=0.3, callbacks=[checkpoint])	