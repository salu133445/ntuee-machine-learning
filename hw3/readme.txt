The script 'train.sh' will generate 7 files:
	4 model files(.json) containing the structure of the trained models
	4 weights files(.hdf5) containing the weights of corresponding models
	1 txt file containg a number which indicates the best trained model.
The scipt 'test.sh' will read the txt file 'best_round.txt' to know which model to use and load the corresponding model structure and weights. 
