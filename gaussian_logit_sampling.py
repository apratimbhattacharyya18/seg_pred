#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc as smisc
import numpy as np

#from pylab import figure, imshow, savefig
import h5py
from tqdm import tqdm
import random
import threading
import os
import sys

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, merge, Lambda, RepeatVector, BatchNormalization, Activation, Dropout
from keras.layers.merge import concatenate, add
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.engine import Layer

import tensorflow as tf


class GaussianLogitSampler(Layer):

	def __init__(self,**kwargs):
		super(GaussianLogitSampler, self).__init__(**kwargs)

	def call(self, x):
		pred_m = x[:,:,:,:K.shape(x)[-1]/2]
		pred_v = K.relu(x[:,:,:,K.shape(x)[-1]/2:])
		pred_v = K.exp(pred_v) - 1.0;
		pred_m = pred_m + tf.multiply( pred_v, K.random_normal(K.shape(pred_v)) )
		return pred_m

	def compute_output_shape(self, input_shape):
		return (input_shape[0],input_shape[1],input_shape[2],input_shape[3]/2)