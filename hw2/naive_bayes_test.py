import numpy as np
import argparse

# parameters
coef_dim = 57
word_char_coef_dim = 54
other_coef_dim = 3

# parser
parser = argparse.ArgumentParser()
parser.add_argument( "modelName", help='the model used for prediction i.e. mymodel.csv' )
parser.add_argument( "testData", help='file containing testing data i.e. spam_test.csv' )
parser.add_argument( "outputCSV", help='file containing testing data i.e. test_X.csv' )

args = parser.parse_args()


# read training data
testing_feature_matrix = np.genfromtxt( args.testData, delimiter=',', usecols=range(1, coef_dim+1) )
testing_word_char_feature_matrix = testing_feature_matrix[ :, 0:word_char_coef_dim ]
testing_other_feature_matrix = testing_feature_matrix[ :, word_char_coef_dim:coef_dim ]
model_big_lambda = np.genfromtxt( args.modelName, delimiter=',', usecols=range(0, word_char_coef_dim), skip_header = 2 )
file = open( args.modelName, 'r' )
file.readline()
threshold = float(file.readline())	

# preprocessing and initialization
bool_testing_word_char_feature_matrix = ( testing_word_char_feature_matrix > 0 )
output_predection = np.zeros( ( testing_feature_matrix.shape[0], 2 ) , dtype=np.int)
# now_test_big_lambda = np.zeros( word_char_coef_dim )
now_test_log_big_lambda = np.zeros( word_char_coef_dim )

# testing stage
for idx_row in range( testing_feature_matrix.shape[0] ):
	#now_test_big_lambda.fill(1.0)
	#for idx_coef in range( word_char_coef_dim ):
	#	now_test_big_lambda *=  model_big_lambda[ int( bool_testing_word_char_feature_matrix[idx_row, idx_coef] ) ]
	now_test_log_big_lambda.fill(0.0)
	for idx_coef in range( word_char_coef_dim ):
		now_test_log_big_lambda +=  model_big_lambda[ int( bool_testing_word_char_feature_matrix[idx_row, idx_coef] ) ]
	now_big_pi_big_lambda = np.prod( now_test_log_big_lambda )
	output_predection[idx_row, 0] = idx_row+1
	output_predection[idx_row, 1] = 1 if now_big_pi_big_lambda > threshold else 0
		
# write the prediction into a csv file
np.savetxt( args.outputCSV, output_predection, delimiter=",", header="id,label", comments='' ,fmt='%i')			