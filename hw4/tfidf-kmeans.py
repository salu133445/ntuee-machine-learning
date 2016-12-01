# For NTU 2016 Fall Machine Learning Class
# Homework 4: Unsupervised Clustering & Dimensionality Reduction - StackOverflow titles classification
# Author: Herman Dong

import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer

# parser
parser = argparse.ArgumentParser()
parser.add_argument( "dataDirectory", help='directory that contains training/test data' )
parser.add_argument( "outputModel", help='postfix of the output model' )
args = parser.parse_args()

# define parameters
n_svd_components = 20

# read input data
input_series = pd.read_csv( args.dataDirectory + 'title_StackOverflow.txt', delimiter='\n', header=None, squeeze=True )

# clean the input data by NLTK's stop word list and Porter Stemmer
porter = PorterStemmer()
train_data = input_series.str.replace( '[^a-zA-Z]', ' ' ).str.replace( ' +', ' ' ).str.lower()
stop_words = set( stopwords.words( 'english' ) )
train_data = train_data.apply( lambda x: [porter.stem(w) for w in wordpunct_tokenize(x) if w not in stop_words] ).str.join(" ")

# TFIDF
stop_words_added = ['use','file', 'get', 'set', 'problem', 'one', 'error', 'list', 'function', 'creat', 'way', 'best']
vectorizer = TfidfVectorizer( analyzer = "word", tokenizer = None, stop_words = stop_words_added, max_df=0.4, min_df=2, max_features = None, norm=u'l2', use_idf=True, smooth_idf = True, sublinear_tf=False )
train_data_features = vectorizer.fit_transform( train_data )
print( "Actual number of TFIDF features: %d" % train_data_features.get_shape()[1] )

# LSA
svd = TruncatedSVD( n_svd_components )
lsa = make_pipeline( svd, Normalizer(copy=False) )
train_data_lsa = lsa.fit_transform( train_data_features )

# K-Means clustering
clustering = KMeans( n_clusters=75, init='k-means++', max_iter=300, tol=0.0001, 
												precompute_distances='auto', verbose=0, random_state=None,
												copy_x=True, n_jobs=1 )													
predictions = clustering.fit_predict( train_data_lsa[:, 1:19] )

# read test data and write predictions to a file
test_pairs = np.genfromtxt( args.dataDirectory + 'check_index.csv', delimiter=',', skip_header=1 ).astype(int)
output = ( predictions[test_pairs[:,1].reshape(-1)] == predictions[test_pairs[:,2].reshape(-1)] ).astype(int)
output_to_file = np.concatenate( (np.arange(5000000).reshape(-1,1), output.reshape(-1,1)), axis = 1 )
np.savetxt( args.outputModel, output_to_file, delimiter=',', header="ID,Ans", comments='', fmt='%i' )