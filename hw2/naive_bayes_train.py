import numpy as np
import argparse

# parameters
coef_dim = 57
word_char_coef_dim = 54
other_coef_dim = 3

# parser
parser = argparse.ArgumentParser()
parser.add_argument( "trainingData", help='file containing training data i.e. spam_train.csv' )
parser.add_argument( "outputModel", help='the trained model i.e. mymodel.csv' )
parser.add_argument( "naiveLambda", type=float, help='threshold parameter of naive bayes classification' )

args = parser.parse_args()


# read training data
input_matrix = np.genfromtxt( args.trainingData, delimiter=',', usecols=range(1, coef_dim+2) )
training_word_char_feature_matrix = input_matrix [ :, 0:word_char_coef_dim ]
training_other_feature_matrix = input_matrix [ :, word_char_coef_dim:coef_dim ]
training_label_matrix = input_matrix [ :, coef_dim ]

# preprocessing
bool_training_word_char_feature_matrix = (training_word_char_feature_matrix > 0)

# training stage
N_spam = training_label_matrix.sum()
N_not_spam = training_label_matrix.shape[0] - training_label_matrix.sum()
P_spam = N_spam/training_label_matrix.shape[0]
threshold = args.naiveLambda * P_spam / ( 1.0 - P_spam )
big_lambda = np.zeros( ( 2, word_char_coef_dim ) )
log_big_lambda = np.zeros( ( 2, word_char_coef_dim ) )

for idx_coef in range( word_char_coef_dim ):
	P_x_true_given_spam = ( ( bool_training_word_char_feature_matrix[:, idx_coef] > 0 ) * ( training_label_matrix > 0 ) ).sum() / N_spam
	P_x_true_given_not_spam =  ( ( bool_training_word_char_feature_matrix[:, idx_coef] > 0 ) * ( training_label_matrix == 0 ) ).sum() / N_not_spam
	# big_lambda[1, idx_coef] = P_x_true_given_spam / P_x_true_given_not_spam
	log_big_lambda[1, idx_coef] = np.log( P_x_true_given_spam / P_x_true_given_not_spam )
	P_x_false_given_spam = ( ( bool_training_word_char_feature_matrix[:, idx_coef] == 0 ) * ( training_label_matrix > 0 ) ).sum() / N_spam
	P_x_false_given_not_spam = ( ( bool_training_word_char_feature_matrix[:, idx_coef] == 0 ) * ( training_label_matrix == 0 ) ).sum() / N_not_spam
	# big_lambda[0, idx_coef] = P_x_false_given_spam / P_x_false_given_not_spam
	log_big_lambda[0, idx_coef] = np.log( P_x_false_given_spam / P_x_false_given_not_spam )
	
# write the trained model into a csv file
np.savetxt( args.outputModel, log_big_lambda, delimiter=",", header="# Model: Naive Bayes Classifier\n"+"{:.16f}".format(threshold), comments='')