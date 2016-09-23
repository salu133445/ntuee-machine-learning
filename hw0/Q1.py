import numpy as np
import argparse

outputFile = 'ans1.txt'

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int)
parser.add_argument("inputFile")

args = parser.parse_args()

matrix = np.loadtxt(args.inputFile)
array = matrix[:, args.index]
array = np.sort(array)

f = open(outputFile, 'w')
f.truncate()

for idx in range(len(array)):
	f.write(str(array[idx]))
	if idx != len(array) - 1:
		f.write(',')