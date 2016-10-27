import numpy as np
import argparse

# parameters
coef_dim = 57

# parser
parser = argparse.ArgumentParser()
parser.add_argument( "modelName", help='the model used for prediction i.e. model' )
parser.add_argument( "testData", help='file containing testing data i.e. spam_test.csv' )
parser.add_argument( "outputCSV", help='file containing testing data i.e. test_X.csv' )
args = parser.parse_args()

# read the model used for prediction
testing_feature_matrix = np.genfromtxt( args.testData, delimiter=',', usecols=range( 1, coef_dim+1 ) )
testing_feature_matrix = np.nan_to_num( testing_feature_matrix )
model_coef_weight_bias = np.genfromtxt( args.modelName+".csv", delimiter=',', skip_header=1)
model_coef_weight_bias = np.nan_to_num( model_coef_weight_bias )

with open( args.modelName+".csv", 'r' ) as file:
   first_line = file.readline()
if first_line[2] == 'N':
	norm = 1
else:
	norm = 0

# test stage
if norm == 1:
	testing_feature_mean = np.zeros( coef_dim )
	testing_feature_standard_deviation = np.zeros( coef_dim )
	for i in range( coef_dim ):
		testing_feature_mean[i] = np.sum( testing_feature_matrix[ :, i ] ) / testing_feature_matrix.shape[0]
		testing_feature_standard_deviation[i] = np.sum( ( testing_feature_matrix[ :, i ] - testing_feature_mean[i] )**2 )
		testing_feature_standard_deviation[i] = ( testing_feature_standard_deviation[i] / testing_feature_matrix.shape[0] )**0.5
		testing_feature_matrix[ :, i ] = ( testing_feature_matrix[ :, i ] - testing_feature_mean[i] ) / testing_feature_standard_deviation[i]

output_predection = np.zeros( ( testing_feature_matrix.shape[0], 2 ) , dtype=np.int)
for count_row in range( testing_feature_matrix.shape[0] ):
	z = np.sum( testing_feature_matrix[ count_row, : ] * model_coef_weight_bias[ 0:coef_dim ] ) + model_coef_weight_bias[coef_dim]
	sigmoid_z = 1 / ( 1 + np.exp( -z ) )
	output_predection[count_row, 0] = count_row+1
	output_predection[count_row, 1] = 1 if sigmoid_z > 0.5 else 0

# write the prediction into a csv file
np.savetxt( args.outputCSV, output_predection, delimiter=",", header="id,label", comments='' ,fmt='%i')