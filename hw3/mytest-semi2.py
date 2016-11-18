# For NTU 2016 Fall Machine Learning Class
# Homework 3: Semi-supervised photo classifier using CIFAR-10 database
# Program 3: semi-supervised photo classifier by K-means clustering using CNN auto-encoder
# Author: Herman Dong
# Tested under Python 2.7.9 with Keras 1.1.1 and sklearn 0.0 in Linux environment

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
from sklearn.cluster import KMeans, MiniBatchKMeans
set_image_dim_ordering('th') #using (samples, channels, width, height)

""" parser """

parser = argparse.ArgumentParser()
parser.add_argument( "dataDirectory", help='directory that contains training/test data' )
parser.add_argument( "inputModel", help='postfix of the output model' )
parser.add_argument( "predictionFileName", help='file name of the output file, i.e. prediction.csv' )
args = parser.parse_args()
postfix = args.inputModel

""" preprocessing stage"""

# define parameter
nb_classes = 10
train_size_labelled = 5000
train_size_unlabelled = 45000
test_size = 10000

# read data
load_test = pickle.load( open( args.dataDirectory + 'test.p', 'rb' ) )
x_test = np.asarray(load_test['data']).reshape( test_size, 3, 32, 32 )
x_id = np.asarray(load_test['ID'])
all_label = pickle.load( open( args.dataDirectory + 'all_label.p', 'rb' ) )
x_train = np.asarray( all_label ).reshape((train_size_labelled, 3, 32, 32))

# load auto-encoder
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
autoencoder.load_weights('weights_' + postfix + '.hdf5')
print('Loaded model from disk')
encoder.summary()

# find initial centroid
encoded_x_train = encoder.predict(x_train).reshape(5000,4096)
n_features = 4096
initial_centroids = np.zeros(( nb_classes, 4096 ))
for i in range(10):
	for j in range(4096):
		initial_centroids[i, j]  = encoded_x_train[i*500:(i+1)*500, j].sum()  / 500

# clustering
encoded_x_test = encoder.predict(x_test)
encoded_x_test = encoded_x_test.reshape( test_size, -1 )
clustering = KMeans( n_clusters=10, init=initial_centroids, n_init=1, max_iter=300, tol=0.0001, 
												precompute_distances='auto', verbose=0, random_state=None,
												copy_x=True, n_jobs=1)
														
predictions = clustering.fit_predict(encoded_x_test)

# save predictions
predictions = np.concatenate((x_id.reshape(-1,1), predictions.reshape(-1,1)), axis=1)
np.savetxt( args.predictionFileName, predictions, delimiter=",", header="ID,class", comments='' ,fmt='%i')
print("Saved prediction to disk")