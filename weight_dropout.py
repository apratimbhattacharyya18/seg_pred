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
import cv2

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, merge, Lambda, RepeatVector, BatchNormalization, Activation, Dropout
from keras.layers.merge import concatenate, add
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras import backend as K

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine import Layer
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy import interfaces

import tensorflow as tf

class Conv2D_wd(Layer):

	def __init__(self,
				 filters,
				 kernel_size,
				 strides=1,
				 padding='valid',
				 data_format=None,
				 dilation_rate=1,
				 activation=None,
				 use_bias=True,
				 kernel_initializer='glorot_uniform',
				 bias_initializer='zeros',
				 kernel_regularizer=None,
				 bias_regularizer=None,
				 activity_regularizer=None,
				 kernel_constraint=None,
				 bias_constraint=None,
				 kl_lambda = 1e-6,
				 drop_prob = 0.20,
				 batch_size = 8,
				 **kwargs):
		super(Conv2D_wd, self).__init__(**kwargs)
		self.rank = 2
		self.filters = filters
		self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
		self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
		self.padding = conv_utils.normalize_padding(padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
		self.activation = activations.get(activation)
		self.use_bias = use_bias
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		self.activity_regularizer = regularizers.get(activity_regularizer)
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		self.kl_lambda = kl_lambda
		self.drop_prob = drop_prob
		self.batch_size = batch_size
		#self.input_spec = InputSpec(ndim=2 + 2)

	def build(self, input_shape):
		if isinstance(input_shape, (list,)):
			input_shape = input_shape[0]

		if self.data_format == 'channels_first':
			channel_axis = 1
		else:
			channel_axis = -1
		if input_shape[channel_axis] is None:
			raise ValueError('The channel dimension of the inputs '
							 'should be defined. Found `None`.')
		input_dim = input_shape[channel_axis]
		self.kernel_shape = self.kernel_size + (input_dim, self.filters)

		self.kernel = self.add_weight(shape=self.kernel_shape,
									  initializer=self.kernel_initializer,
									  name='kernel',
									  regularizer=self.kernel_regularizer,
									  constraint=self.kernel_constraint)

		if self.use_bias:
			self.bias = self.add_weight(shape=(self.filters,),
										initializer=self.bias_initializer,
										name='bias',
										regularizer=self.bias_regularizer,
										constraint=self.bias_constraint)
		else:
			self.bias = None
		# Set input spec.
		#self.input_spec = InputSpec(ndim=self.rank + 2, axes={channel_axis: input_dim})
		self.built = True
		kernel_reg = 1e-8 * K.sum(K.square(self.kernel)) #
		bias_reg = 1e-8 * K.sum(K.square(self.bias))
		self.add_loss(kernel_reg)
		self.add_loss(bias_reg)


	def sample_mask_conv(self,p):
		eps = tf.multiply(K.constant( 1e-30,dtype=tf.float32),K.ones((self.batch_size,self.kernel_shape[0],self.kernel_shape[1],self.kernel_shape[2],self.kernel_shape[3]), dtype=tf.float32)) 
		temp = 0.01
		_ones  = K.ones((self.batch_size,self.kernel_shape[0],self.kernel_shape[1],self.kernel_shape[2],self.kernel_shape[3]), dtype=tf.float32);

		unif_noise = K.random_uniform(shape=(self.batch_size,self.kernel_shape[0],self.kernel_shape[1],self.kernel_shape[2],self.kernel_shape[3]),dtype=tf.float32)
		drop_prob_mask = K.log(p + eps)  - K.log(1. - p + eps) + K.log(unif_noise + eps) - K.log( _ones - unif_noise + eps)#- K.log(1. - p + eps) + K.log(unif_noise + eps)
		drop_prob_mask = K.sigmoid(drop_prob_mask / temp)
		mask = 1. - drop_prob_mask
		return mask

	def sample_mask_bias(self,p):
		eps = tf.multiply(K.constant( 1e-30,dtype=tf.float32),K.ones((self.batch_size,self.kernel_shape[3]), dtype=tf.float32)) 
		temp = 0.01
		_ones  = K.ones((self.batch_size,self.kernel_shape[3]), dtype=tf.float32);

		unif_noise = K.random_uniform(shape=(self.batch_size,self.kernel_shape[3]),dtype=tf.float32)
		drop_prob_mask = K.log(p + eps)  - K.log(1. - p + eps) + K.log(unif_noise + eps) - K.log( _ones - unif_noise + eps)#- K.log(1. - p + eps) + K.log(unif_noise + eps)
		drop_prob_mask = K.sigmoid(drop_prob_mask / temp)
		mask = 1. - drop_prob_mask
		return mask

	def call(self, _inputs):

		mask_c = None;
		mask_b = None;

		inputs = _inputs
		mask_p = K.sigmoid(K.zeros((self.batch_size,self.kernel_shape[0],self.kernel_shape[1],self.kernel_shape[2],self.kernel_shape[3]), dtype=tf.float32)/1.0 + K.log(0.2/0.8));
		mask_c = self.sample_mask_conv(mask_p);
		mask_b = self.sample_mask_bias(tf.multiply(K.constant(self.drop_prob,dtype=tf.float32),K.ones((self.batch_size,self.kernel_shape[3]), dtype=tf.float32)))


		masked_k = K.repeat_elements(K.expand_dims(self.kernel,axis=0),self.batch_size,axis=0)
		masked_k = tf.multiply(masked_k,mask_c);

		masked_b = K.repeat_elements(K.expand_dims(self.bias,axis=0),self.batch_size,axis=0)
		masked_b = tf.multiply(masked_b,mask_b);	
		
		def single_conv(tupl):
			x, kernel = tupl
			return K.conv2d(x, kernel, strides=self.strides, padding=self.padding)

		def single_bias_add(tupl):
			x, _bias = tupl
			return K.bias_add( x, _bias, data_format=self.data_format)	
		
		outputs = tf.squeeze(tf.map_fn( single_conv, (tf.expand_dims(inputs, 1), masked_k), dtype=tf.float32), axis=1 )


		if self.use_bias:
			outputs = tf.squeeze(tf.map_fn( single_bias_add, (tf.expand_dims(outputs, 1), masked_b), dtype=tf.float32), axis=1 )

		if self.activation is not None:
			return self.activation(outputs)
		return outputs

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_last':
			space = input_shape[1:-1]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(
					space[i],
					self.kernel_size[i],
					padding=self.padding,
					stride=self.strides[i],
					dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0],) + tuple(new_space) + (self.filters,)
		if self.data_format == 'channels_first':
			space = input_shape[2:]
			new_space = []
			for i in range(len(space)):
				new_dim = conv_utils.conv_output_length(
					space[i],
					self.kernel_size[i],
					padding=self.padding,
					stride=self.strides[i],
					dilation=self.dilation_rate[i])
				new_space.append(new_dim)
			return (input_shape[0], self.filters) + tuple(new_space)

	def get_config(self):
		config = {
			'rank': self.rank,
			'filters': self.filters,
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			'padding': self.padding,
			'data_format': self.data_format,
			'dilation_rate': self.dilation_rate,
			'activation': activations.serialize(self.activation),
			'use_bias': self.use_bias,
			'kernel_initializer': initializers.serialize(self.kernel_initializer),
			'bias_initializer': initializers.serialize(self.bias_initializer),
			'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
			'bias_regularizer': regularizers.serialize(self.bias_regularizer),
			'activity_regularizer': regularizers.serialize(self.activity_regularizer),
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
			'bias_constraint': constraints.serialize(self.bias_constraint)
		}
		base_config = super(Conv2D_wd, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))