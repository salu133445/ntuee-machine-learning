import numpy as np
import argparse

# Usage:  pm25.py trainFile testFile iteration learingRate regCoef method --norm --wCB
#    trainFile      file containing training data i.e. train.csv
#    testFile       file containing testing data i.e. test_X.csv
#    iteration      number of iterations
#    learningRate   learning rate of linear regression
#    regCoef        coefficient lambda for regularization
#    method         0-Vanilla GD, 1-AdaGrad, 2-AdaDelta
#    --norm         1-Yes, 0-No: normalize training data while training
#    --wCB          1-Yes, 0-No: write coefficient and bias into a file named "coef_iteration_regcoef_method.csv"
# Output: "ans_iteration_regcoef_method.csv" containing prediction result of testing file


# parameters
outputFile = 'ans.csv'
coef_dim, train_size = 18, 9
ada_smoothTerm = 0.000000001 # avoid divided-by-zero condition for adaGrad and adaDelta
adaGradDelta_coef = 0.95

# parser
parser = argparse.ArgumentParser()
parser.add_argument("trainFile", help='file containing training data i.e. train.csv')
parser.add_argument("testFile", help='file containing testing data i.e. test_X.csv')
parser.add_argument("iteration", type=int, help='number of iterations')
parser.add_argument("learingRate", type=float, help='learning rate of linear regression')
parser.add_argument("regCoef", type=float, help='coefficient lambda for regularization ')
parser.add_argument("method", type=int, help='0-Vanilla GD, 1-AdaGrad, 2-AdaDelta')
parser.add_argument("--norm", type=bool, help='normalize training data while training')
parser.add_argument("--wcb", type=bool, help='0-N, 1-Y: write coefficient and bias into a file named "coef_iteration_regcoef_method.csv"')
args = parser.parse_args()


# initialization
matrix = np.genfromtxt(args.trainFile, delimiter=',', skip_header=1, usecols=range(3,27))
matrix = np.nan_to_num(matrix)

count_iter = 0

trainSet = np.zeros(shape=(12,coef_dim,24*20))
for i in range(12):
	for j in range(20):
		trainSet[i,:,24*j:24*(j+1)] = matrix[(20*18*i+18*j):(20*18*i+18*(j+1)),:]
bias = 0.0
coef = 2*np.random.random_sample((coef_dim,train_size))-1

part_l_w = np.empty(shape=(coef_dim,train_size))
part_l_w_sqAccu = np.zeros(shape=(coef_dim,train_size))
part_l_b_sqAccu = 0.0
adaDelta_runningAveg_w = np.zeros(shape=(coef_dim,train_size))
adaDelta_runningAveg_bias = 0.0
w_delta_runningAveg = np.zeros(shape=(coef_dim,train_size))
bias_delta_runningAveg = 0.0

#Normalization 
if args.norm:
	trainSet_mean = np.zeros(coef_dim)
	trainSet_stdev = np.zeros(coef_dim)
	mod_trainSet = np.zeros(shape=(12,coef_dim,24*20))
	for i in range(12):
		for j in range(coef_dim):
			trainSet_mean[j] += np.sum(trainSet[i,j,:])
	trainSet_mean = trainSet_mean / (12*trainSet.shape[2])
	for i in range(12):
		for j in range(coef_dim):
			trainSet_stdev[j] += np.sum((trainSet[i,j,:]-trainSet_mean[j])**2)
	trainSet_stdev = (trainSet_stdev/(12*trainSet.shape[2]))**0.5
	for i in range(12):
		for j in range(coef_dim):
			mod_trainSet[i,j,:] = (trainSet[i,j,:]-trainSet_mean[j])/trainSet_stdev[j]
else:
	mod_trainSet = trainSet


# training stage
for count_iter in range(args.iteration):
	loss = 0.0
	part_l_b = 0.0
	part_l_w.fill(0.0)
	for i in range(12):
		count_col = count_iter%(train_size+1)
		while count_col+train_size+1<trainSet.shape[2]:
			temp_sum = np.sum(coef*mod_trainSet[i,:,count_col:(count_col+train_size)])+bias
			temp_sum = 2*(temp_sum-mod_trainSet[i,9,count_col+train_size+1])
			#if count_iter%10==0:
			#	temp_sum_2 = temp_sum - mod_trainSet[i,9,count_col+train_size+1]
			#	temp_sum_2 = temp_sum_2**2
			#	loss += temp_sum_2
			part_l_w += mod_trainSet[i,:,count_col:(count_col+train_size)]*temp_sum
			part_l_w += coef*(2*args.regCoef)
			part_l_b += temp_sum
			count_col += 1
	#if count_iter%10==0:
	#	print(count_iter, "MSE: ", loss/5652)		
	
	# update coefficient and bias
	if args.method == 0:
		coef -= part_l_w*args.learingRate
		bias -= part_l_b*args.learingRate
	elif args.method == 1:
		part_l_w_sqAccu += part_l_w**2
		part_l_b_sqAccu += part_l_b**2
		coef -= part_l_w/(np.sqrt(part_l_w_sqAccu)+ada_smoothTerm)*args.learingRate
		bias -= part_l_b/(np.sqrt(part_l_b_sqAccu)+ada_smoothTerm)*args.learingRate
	elif args.method == 2:
		adaDelta_runningAveg_w = adaGradDelta_coef*adaDelta_runningAveg_w + (1-adaGradDelta_coef)*(part_l_w**2)
		adaDelta_runningAveg_bias = adaGradDelta_coef*adaDelta_runningAveg_bias + (1-adaGradDelta_coef)*(part_l_b**2)
		w_delta = part_l_w*(np.sqrt(w_delta_runningAveg+ada_smoothTerm)/np.sqrt(adaDelta_runningAveg_w+ada_smoothTerm))
		bias_delta = part_l_b*(np.sqrt(bias_delta_runningAveg+ada_smoothTerm)/np.sqrt(adaDelta_runningAveg_bias+ada_smoothTerm))
		coef -= w_delta
		bias -= bias_delta
		w_delta_runningAveg = adaGradDelta_coef*w_delta_runningAveg + (1-adaGradDelta_coef)*(w_delta**2)
		bias_delta_runningAveg = adaGradDelta_coef*bias_delta_runningAveg + (1-adaGradDelta_coef)*(bias_delta**2)

if args.wCB:
	np.savetxt('coef_'+str(args.iteration)+'_'+str(args.learingRate)+'_'+str(args.regCoef)+'.csv', coef, delimiter=",")
	file = open('bias_'+str(args.iteration)+'_'+str(args.learingRate)+'_'+str(args.regCoef)+'.csv', 'w', encoding = 'UTF-8')
	file.truncate()
	file.write(str(bias))
	file.close()


# testing stage

matrix_test = np.genfromtxt(args.testFile, delimiter=',', usecols=range(2,11))
matrix_test = np.nan_to_num(matrix_test)

if arg.norm:
	for i in range(matrix_test.shape[0]):
		matrix_test[i,:] = (matrix_test[i,:]-trainSet_mean[i%18])/trainSet_stdev[i%18]

output = np.array([["id","value"]])
count_row = 0
while count_row+coef_dim-1 <= matrix.shape[0]:
	if args.norm:
		output = np.append(output, [["id_"+str(int(count_row/coef_dim)), '{:.10f}'.format((np.sum(coef*matrix_test[count_row:count_row+18,:])+bias)*trainSet_stdev[9]+trainSet_mean[9])]], axis=0)
	else:
		output = np.append(output, [["id_"+str(int(count_row/coef_dim)), '{:.10f}'.format(np.sum(coef*matrix_test[count_row:count_row+18,:])+bias)]], axis=0)
	count_row += coef_dim
np.savetxt(outputFile, output, delimiter=",", fmt="%s")
