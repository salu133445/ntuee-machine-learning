# For NTU 2016 Fall Machine Learning Class
# Homework 3: Semi-supervised photo classifier using CIFAR-10 database
# Program 1: supervised photo classifier by CNN
# Author: Herman Dong
# Tested under Python 2.7.9 with Keras 1.1.1 in Linux environment

import argparse
import numpy as np
import pickle
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.noise import GaussianNoise
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2, l1l2

""" parser """

parser = argparse.ArgumentParser()
parser.add_argument( "dataDirectory", help='directory that contains training/test data' )
parser.add_argument( "inputModel", help='postfix of the output model' )
parser.add_argument( "predictionFileName", help='file name of the output file, i.e. prediction.csv' )
args = parser.parse_args()
postfix = args.inputModel

""" preprocessing stage """

# define parameters
nb_classes = 10
test_size = 10000

# read data
with open('best_round.txt') as file:
    best_round = file.read()
now_postfix = postfix + '-' + str(best_round)

load_test = pickle.load( open( args.dataDirectory + 'test.p', 'rb' ) )
x_test = np.asarray(load_test['data']).reshape( test_size, 3, 32, 32 )
x_id = np.asarray(load_test['ID'])

# load model
json_file = open('model_' + now_postfix + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('weights_' + now_postfix + '.hdf5')
print('Loaded model from disk')

# testing stage
predictions = model.predict_classes( x_test )
predictions = np.concatenate((x_id.reshape(-1,1), predictions.reshape(-1,1)), axis=1)
np.savetxt( args.predictionFileName, predictions, delimiter=",", header="ID,class", comments='' ,fmt='%i')
print("\nSaved prediction to disk")