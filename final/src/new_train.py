
# coding: utf-8

# In[1]:

import argparse
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import minmax_scale
from keras.wrappers.scikit_learn import KerasRegressor


# In[2]:
max_rows = None
""" Numercial Data Preprocessing """
print('Reading Numercial Data')
train_data_numercial = pd.read_csv('train', delimiter=',', header=None, dtype=float, usecols=([0]+list(range(4,19))+list(range(20,41))), nrows=max_rows)
test_data_numercial = pd.read_csv('test.in', delimiter=',', header=None, dtype=float, usecols=([0]+list(range(4,19))+list(range(20,41))), nrows=max_rows)
print('Normalizing Numercial Data')
all_numercial_normalized = np.concatenate((train_data_numercial.values, test_data_numercial.values ), axis=0)
all_numercial_normalized = minmax_scale(all_numercial_normalized, feature_range=(0, 1), copy=True, axis=0)

train_data_numercial[:] = all_numercial_normalized[0:len(train_data_numercial), :]
test_data_numercial[:] = all_numercial_normalized[len(train_data_numercial):, :]


# In[3]:

""" Categorical Data Preprocessing """
print('Reading Categorical Data')

train_data_categorical = pd.read_csv('train', delimiter=',', header=None, usecols=[1,2,3], dtype=str, names = [ 'protocol_type', 'service', 'flag'], nrows=max_rows )
test_data_categorical = pd.read_csv('test.in', delimiter=',', header=None, usecols=[1,2,3], dtype=str, names = [ 'protocol_type', 'service', 'flag'], nrows=max_rows )

print('Expanding Categorical Data')
all_categorical = pd.concat( (train_data_categorical, test_data_categorical), axis=0 )

train_data_expand = train_data_numercial
test_data_expand = test_data_numercial

for col_name in all_categorical.columns.values:
	to_append = pd.get_dummies( all_categorical[col_name], prefix=col_name, sparse=False ).astype(bool)
	train_data_expand = pd.concat( (train_data_expand, to_append[0:len(train_data_expand)]), axis=1 )
	test_data_expand = pd.concat( (test_data_expand, to_append[len(train_data_expand):]), axis=1 )


# In[4]:

""" Label Data """
print('Reading Label Data')
attack_expand_type_names = ['normal', 'u2r', 'r2l', 'probe', 'apache2', 'back', 'mailbomb', 'processtable', 'snmpgetattack', 'teardrop', 'smurf', 'land', 'neptune', 'pod', 'udpstorm']
attack_expand_type_dict = {'normal':0}
attack_dos_subtype_names = ['apache2', 'back', 'mailbomb', 'processtable', 'snmpgetattack', 'teardrop', 'smurf', 'land', 'neptune', 'pod', 'udpstorm']
with open( 'training_attack_types.txt', 'r' ) as file:
	for line in file.readlines():
		if line.strip().split(' ')[1] == 'dos':
			attack_expand_type_dict[line.split(' ')[0]] = attack_expand_type_names.index(line.strip().split(' ')[0])
		else:
			attack_expand_type_dict[line.split(' ')[0]] = attack_expand_type_names.index(line.strip().split(' ')[1])

attack_type_names = [ 'normal', 'dos', 'u2r', 'r2l', 'probe' ]
attack_type_dict = {'normal':0, 'u2r':2, 'r2l':3, 'probe':4}
for subtype_name in attack_dos_subtype_names:
	attack_type_dict[subtype_name] = 1


attack_txt =pd.read_csv('training_attack_types.txt',delimiter=' ',names=['subtype','type'],dtype=str)


#attack_names = pd.DataFrame(attack_expand_type_names)
#attack_names_dict = 
#attack_dos_names = pd.DataFrame(attack_dos_subtype_names)

#TODO the converter seems wrong
#train_labels2 = pd.read_csv('train', delimiter=',', header=None, usecols=[41], names=['labels'], converters={41: lambda s: attack_txt.subtype[s[:-1]]}, nrows=max_rows)
train_labels = pd.read_csv('train', delimiter=',', header=None, usecols=[41], names=['labels'], converters={41: lambda s: attack_expand_type_dict[s[:-1]]}, nrows=max_rows)

train_labels_expand = pd.get_dummies( train_labels['labels'], prefix='labels', sparse=False ).astype(bool)


# In[6]:

""" Training Stage """
print('Training Model')
model = Sequential()
model.add( Dense( train_data_expand.shape[1], input_dim=train_data_expand.shape[1], init='normal', activation='relu' ) )
model.add( Dense( 500, init='normal', activation='relu' ) )
model.add(Dropout(0.5))
model.add( Dense( 800, init='normal', activation='relu' ) )
model.add(Dropout(0.5))
model.add( Dense( 250, init='normal', activation='relu' ) )
model.add(Dropout(0.5))
model.add( Dense( 100, init='normal', activation='relu' ) )
model.add(Dropout(0.5))
model.add( Dense( train_labels_expand.shape[1], init='normal', activation='sigmoid' ) )
model.compile( loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'] )
model.summary()


# In[8]:

shuffled_index = np.random.shuffle(np.arange(train_data_expand.shape[0]))
shuffled_X = train_data_expand.as_matrix()[shuffled_index].reshape(train_data_expand.shape[0], train_data_expand.shape[1])
shuffled_Y = train_labels_expand.as_matrix()[shuffled_index].reshape(train_labels_expand.shape[0], train_labels_expand.shape[1])


#shuffled_X_Y = np.concatenate( ( train_data_expand.values, train_labels.values.reshape(-1,1) ), axis=1 )
#shuffled_X_Y = np.random.shuffle( shuffled_X_Y )


validation_split = 0.25
validation_cut_point = (1-validation_split)*shuffled_X.shape[0]
val_X = shuffled_X[validation_cut_point:]
val_Y = shuffled_Y[validation_cut_point:]
shuffled_X = shuffled_X[:validation_cut_point]
shuffled_Y = shuffled_Y[:validation_cut_point]

checkpoint = ModelCheckpoint('mymodel.h5', monitor='val_loss', save_best_only=True, mode='auto')
model.fit(shuffled_X, shuffled_Y, nb_epoch=60, batch_size=5000, validation_data=(val_X, val_Y), callbacks=[checkpoint] )

"""
model.load( 'mymodel.h5' )

val_predictions = model.predict_classes(val_X).reshape(-1,1)
val_errors = ( val_predictions != val_Y )
val_Y_pred_err = np.concatenate((val_Y, val_predictions, val_errors), axis=1)
val_Y_pred_err = val_Y_pred_err[val_Y_pred_err[:,2]==1]

error_table = np.zeros((6,6))
error_table[0,1:] = np.arange(5)
error_table[1:,0] = np.arange(5)
# Y in row and predictions in column
for i in range(5):
	for j in range(5):
		error_table[i+1, j+1] = ((val_Y_pred_err[:, 0]==i)*(val_Y_pred_err[:, 10]==j)).sum()
"""

# In[9]:
'''
""" Testing Stage """
print('Testing Stage')
model.load( 'mymodel.h5' ) # since I use ModelCheckpoint 'save_best_only=True'
'''
X_test = test_data_expand.as_matrix()
expand_predictions = model.predict_classes(X_test)
expand_predictions.reshape(-1,1)
print(expand_predictions[0])
# predictions = attack_type_dict[attack_expand_type_names[expand_predictions]]
predictions = np.zeros((expand_predictions.shape[0], 1))
for i in range(expand_predictions.shape[0]):
	predictions[i] = attack_type_dict[attack_expand_type_names[expand_predictions[i]]];
# predictions = np.apply_along_axis(lambda k: attack_type_dict[attack_expand_type_names[k[0]]], 1, expand_predictions)
# predictions = np.apply_along_axis(lambda k: 2*k, 0, expand_predictions)
print(predictions.shape)
# In[15]:

write_to_file = np.column_stack(((np.arange(predictions.shape[0])+1), predictions))
np.savetxt('predictions.csv', write_to_file, delimiter=",", header="id,label", comments='' ,fmt='%i')
print('Saved predictions to disk')


# In[ ]:



