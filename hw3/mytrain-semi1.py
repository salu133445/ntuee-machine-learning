# For NTU 2016 Fall Machine Learning Class
# Homework 3: Semi-supervised photo classifier using CIFAR-10 database
# Program 2: semi-supervised photo classifier by CNN using self-training
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
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2, l1l2
from keras.backend import set_image_dim_ordering
set_image_dim_ordering('th') #using (samples, channels, width, height)

""" parser """

parser = argparse.ArgumentParser()
parser.add_argument( "dataDirectory", help='directory that contains training/test data' )
parser.add_argument( "outputModel", help='postfix of the output model' )
parser.add_argument( "predictionFileName", help='file name of the output file, i.e. prediction.csv' )
args = parser.parse_args()
postfix = args.outputModel

""" preprocessing stage"""

# define parameter
nb_classes = 10
train_size_labelled = 5000
train_size_unlabelled = 45000
test_size = 10000
iterations = 3
validation_split = 0.3
supervised_epoch = 100
semi_supervised_epoch = 75

# create model
model = Sequential()
model.add(Convolution2D(32,3,3, input_shape=(3,32,32), border_mode='same'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Convolution2D(64,3,3, border_mode='same'))
model.add(MaxPooling2D((2,2)))
model.add(Convolution2D(64,3,3, border_mode='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

# read train data
all_labelled = pickle.load( open( args.dataDirectory + 'all_label.p', 'rb' ) )
all_unlabelled = pickle.load( open( args.dataDirectory + 'all_unlabel.p', 'rb' ) )

# pre-process the data
x_train = np.asarray( all_labelled ).reshape( train_size_labelled, 3072 )
y_labels = np.asarray([ [i]*500 for i in range(nb_classes) ], dtype=int).reshape((-1,1))
shuffled_x_train_y_labels = np.concatenate(( x_train, y_labels ), axis=1)
np.random.shuffle(shuffled_x_train_y_labels)
train_validation_split = int((1-validation_split)*train_size_labelled)
x_train = shuffled_x_train_y_labels[0:train_validation_split, 0:3072].reshape((-1, 3, 32, 32))
x_val = shuffled_x_train_y_labels[train_validation_split:train_size_labelled, 0:3072].reshape((-1, 3, 32, 32))
y_labels = shuffled_x_train_y_labels[0:train_validation_split,3072].reshape((-1,1))
y_val_labels = shuffled_x_train_y_labels[train_validation_split:train_size_labelled,3072].reshape((-1,1))
y_val = np_utils.to_categorical( y_val_labels, nb_classes )

x_train_unlabelled = np.asarray( all_unlabelled ).reshape( train_size_unlabelled, 3, 32, 32 )
 
""" training stage """
i_round_best = np.zeros(iterations+1)
# train the labelled set
y_train = np_utils.to_categorical( y_labels, nb_classes )
now_postfix = postfix + '-' + str(0)
weight_saved_path = 'weights_' + now_postfix + '.hdf5'
checkpoint = ModelCheckpoint(weight_saved_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')	
fitting = model.fit( x_train, y_train, batch_size=100, nb_epoch=supervised_epoch, validation_data=(x_val, y_val), callbacks=[checkpoint])

result = model.evaluate( x_train, y_train )
print( "\nFinal Evaluation", result[1] )

# save model structure to JSON file
model_json = model.to_json()
model_saved_path = 'model_' + now_postfix + '.json'
with open(model_saved_path, "w") as json_file:
	 json_file.write(model_json)
print("Saved model to disk")

# save the training history
history_val_acc = np.asarray( fitting.history['val_acc'] )

i_round_best[0] = max(history_val_acc)


""" self training """
count_collected_unlabelled = 0

for i in range(iterations):
	print "Running", i+1, "round of", iterations, "rounds!!!"
	
	# make probabilistic predictions of unlabelled data
	prob_predictions = model.predict_proba( x_train_unlabelled )
	
	# reset the weights of the model
	weights = model.get_weights()
	weights = [np.random.permutation(w) for w in weights]
	model.set_weights(weights)
	
	# collect the index of unlabelled data of enough confidence
	th = 0.99
	collected_unlabel_argmax = np.array([])
	collected_unlabel_index = np.array([], dtype=int)
	prob_predictions_max = np.amax( prob_predictions, axis= 1).reshape(-1)
	prob_predictions_argmax = np.argmax( prob_predictions, axis= 1 ).reshape(-1)
	for j in range( x_train_unlabelled.shape[0] ):
		if  prob_predictions_max[j] > th:
			collected_unlabel_index = np.append(collected_unlabel_index, j )
	count_collected_unlabelled = count_collected_unlabelled + len( collected_unlabel_index )
	print 'threshold: ', th, ', # of collected unlabelled data: ', count_collected_unlabelled
	
	# add collected unlabelled data into training data
	y_labels = np.append( y_labels,  prob_predictions_argmax[collected_unlabel_index])
	x_train = np.concatenate(( x_train, x_train_unlabelled[collected_unlabel_index]), axis=0 )
	
	# remove selected unlabeled data from unlabeled data
	x_train_unlabelled = np.delete( x_train_unlabelled, collected_unlabel_index, axis=0 )
	prob_predictions = np.delete( prob_predictions, collected_unlabel_index, axis=0 )

	# fitting
	y_train = np_utils.to_categorical( y_labels, nb_classes )
	now_postfix = postfix + '-' + str(i+1)
	weight_saved_path = 'weights_' + now_postfix + '.hdf5'
	checkpoint = ModelCheckpoint(weight_saved_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	fitting = model.fit(x_train,y_train,batch_size=100,nb_epoch=semi_supervised_epoch, callbacks=[checkpoint], validation_data=(x_val, y_val))


	# save model structure to JSON file
	model_json = model.to_json()
	model_saved_path = 'model_' + now_postfix + '.json'
	with open(model_saved_path, "w") as json_file:
		 json_file.write(model_json)
	# model.save_weights("mymodel_supervised_4NN_1DL_dropout_half.h5")
	print("Saved model to disk")

	# record the best validation accuracy
	history_val_acc = np.asarray( fitting.history['val_acc'] )
	
	i_round_best[i] = max(history_val_acc)

print('Best model: round ', np.argmax(i_round_best), ' with accuracy = ', np.max(i_round_best))
with open('best_round.txt', 'w') as file:
    file.write(str(np.argmax(i_round_best)))
