import numpy as np
import argparse

# parameters
coef_dim = 57
ada_smoothTerm = 0.000000001 # avoid divided-by-zero condition for adaGrad and adaDelta
adaGradDelta_coef = 0.95

# parser
parser = argparse.ArgumentParser()
parser.add_argument( "trainingData", help='file containing training data i.e. spam_train.csv' )
parser.add_argument( "outputModel", help='the trained model i.e. model' )
parser.add_argument( "iteration", type=int, help='number of iterations' )
parser.add_argument( "learningRate", type=float, help='learning rate of linear regression' )
parser.add_argument( "regCoef", type=float, help='coefficient lambda for regularization ' )
parser.add_argument( "method", type=int, help='0-Vanilla GD, 1-AdaGrad, 2-AdaDelta' )
parser.add_argument( "--norm", type=bool, help='normalize training data while training' )
args = parser.parse_args()

# read training data
input_matrix = np.genfromtxt( args.trainingData, delimiter=',', usecols=range(1, coef_dim+2) )
training_feature_matrix = input_matrix [ :, 0:coef_dim ]
training_label_matrix = input_matrix [ :, coef_dim ]

# initialization
coef_weight_bias = np.zeros( coef_dim + 1 )
partial_loss_weight_bias = np.empty( coef_dim + 1 )
partial_loss_weight_bias_sqAccu = np.zeros( coef_dim + 1 )
adaDelta_runningAveg_weight_bias = np.zeros( coef_dim + 1 )
weight_bias_delta_runningAveg = np.zeros( coef_dim + 1 )

# normalization
if args.norm:
	training_feature_mean = np.zeros( coef_dim )
	training_feature_standard_deviation = np.zeros( coef_dim )
	for i in range( coef_dim ):
		training_feature_mean[i] = np.sum( training_feature_matrix[ :, i ] ) / training_feature_matrix.shape[0]
		training_feature_standard_deviation[i] = np.sum( ( training_feature_matrix[ :, i ] - training_feature_mean[i] )**2 )
		training_feature_standard_deviation[i] = ( training_feature_standard_deviation[i] / training_feature_matrix.shape[0] )**0.5
		training_feature_matrix[ :, i ] = ( training_feature_matrix[ :, i ] - training_feature_mean[i] ) / training_feature_standard_deviation[i]
	
# training stage
for count_iteration in range( args.iteration ):
	#loss = 0.0
	partial_loss_weight_bias_sqAccu.fill(0.0)
	for count_row in range( training_feature_matrix.shape[0] ):
		z = np.sum( training_feature_matrix[ count_row, : ] * coef_weight_bias[ 0:coef_dim ] ) + coef_weight_bias[ coef_dim ]
		sigmoid_z = 1 / ( 1 + np.exp( -z ) )
		#loss -= training_label_matrix[ count_row ] * np.log(sigmoid_z) + ( 1 - training_label_matrix[ count_row ] ) * ( 1 - np.log(sigmoid_z) )
		partial_loss_weight_bias[ 0:coef_dim ] -=  ( training_label_matrix[ count_row ] - sigmoid_z ) * training_feature_matrix[ count_row ]
		partial_loss_weight_bias[ 0:coef_dim ] += 2 * args.regCoef * coef_weight_bias[ 0:coef_dim ]
		partial_loss_weight_bias[ coef_dim ] -=  ( training_label_matrix[ count_row ] - sigmoid_z )
	if args.method == 0:
		coef_weight_bias -= partial_loss_weight_bias*args.learningRate
	elif args.method == 1:
		partial_loss_weight_bias_sqAccu += partial_loss_weight_bias**2
		coef_weight_bias -= partial_loss_weight_bias/(np.sqrt(partial_loss_weight_bias_sqAccu)+ada_smoothTerm)*args.learningRate
	elif args.method == 2:
		adaDelta_runningAveg_weight_bias = adaGradDelta_coef*adaDelta_runningAveg_weight_bias + (1-adaGradDelta_coef)*(partial_loss_weight_bias**2)
		weight_bias_delta = partial_loss_weight_bias*(np.sqrt(weight_bias_delta_runningAveg+ada_smoothTerm)/np.sqrt(adaDelta_runningAveg_weight_bias+ada_smoothTerm))
		coef_weight_bias -= weight_bias_delta
		weight_bias_delta_runningAveg = adaGradDelta_coef*weight_bias_delta_runningAveg + (1-adaGradDelta_coef)*(weight_bias_delta**2)
	#if count_iteration%100==0:
	#	print( "Computing...", 100*count_iteration/args.iteration, "%", "   Loss:", loss)
		
# write the trained model into a csv file
if args.norm:
	np.savetxt( args.outputModel+".csv", coef_weight_bias, delimiter=",", header="Normalized Logistic Regression" )
else:
	np.savetxt( args.outputModel+".csv", coef_weight_bias, delimiter=",", header="Unnormalization Logistic Regression" )