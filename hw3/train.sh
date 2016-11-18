#!/bin/bash
KERAS_BACKEND=theano THEANO_FLAGS='floatX=float32,device=gpu0,lib.cnmem=1' python2 mytrain-semi2.py $1 $2
