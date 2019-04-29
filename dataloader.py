import scipy.io as sio
import scipy.misc as smisc
import numpy as np
import h5py
from tqdm import tqdm
import random
import threading
import os
import sys
import cv2
import math
import time
import pickle

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Reshape, merge, Lambda, RepeatVector, BatchNormalization, Activation, Dropout, Flatten
from keras.layers.merge import concatenate, add
from keras.engine import Layer
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras import backend as K

import tensorflow as tf

class DataLoader:

	def __init__(self, batch_size):
		print('Loading sequence names ...')
		with open('./data/all_im_seq_train.pkl', 'rb') as f:
			self.all_im_seq_train = pickle.load(f)
		with open('./data/all_im_seq_val.pkl', 'rb') as f:
			self.all_im_seq_val = pickle.load(f)    
		print('Done.')

		print('Loading odo data ...')
		with open('./data/train_odo_data.pkl', 'rb') as f:
			self.train_odo_data = pickle.load(f)
		with open('./data/val_odo_data.pkl', 'rb') as f:
			self.val_odo_data = pickle.load(f)    
		print('Done.')

		self.train_data_file = './data/seg_data_train.h5';
		self.val_data_file = './data/seg_data_val.h5';

		with h5py.File( self.train_data_file, 'r') as f:
			self.all_train_keys = f.keys();
		self.all_train_grps = self.all_im_seq_train;
		self.all_train_odo_keys = self.train_odo_data.keys();

		self.batch_size = batch_size

	def train_data_batch(self,):
		data_X_s = [];
		data_X_o = [];
		data_Y = [];
		count = 0;
		with h5py.File( self.train_data_file, 'r') as f:
			while 1:
				grp_idx = random.randint(0,len(self.all_train_grps)-1);
				seq_idx = random.randint(0,24);
				
				curr_seq = self.all_train_grps[grp_idx][seq_idx:seq_idx+5]
				curr_seq =[x.split('/')[-1][0:-4] for x in curr_seq]

				curr_seq_segs = [];
				curr_seq_odo = [];
				for frame in curr_seq:
					if frame in self.all_train_keys and frame in self.all_train_odo_keys:
						curr_seq_segs.append(f[frame][()]);
						curr_seq_odo.append( np.expand_dims(self.train_odo_data[frame], axis=0) );
					else:
						continue;

				if len(curr_seq_segs) < 5:
					continue;

				data_X_s.append( np.concatenate(curr_seq_segs[0:4], axis = -1) )
				data_X_o.append( np.concatenate(curr_seq_odo, axis = 0) )

				data_Y.append( curr_seq_segs[-1] )

				count += 1
				if count == self.batch_size:
					break;

		data_X_s = np.array(data_X_s);
		data_X_o = np.array(data_X_o);
		data_Y = np.array(data_Y);
		return ( data_X_s, data_X_o, data_Y)

	def get_val_data(self,num_of_test_examples,test_samples):
		all_val_grps = self.all_im_seq_val;
		data_X_s = [];
		data_X_o = [];
		data_Y = [];
		data_Y_ = [];
		all_val_odo_keys = self.val_odo_data.keys();
		with h5py.File( self.val_data_file, 'r') as f:
			for i in tqdm(xrange(0,len(all_val_grps))):
				grp_idx = i;
				seq_idx = 20 - 5;
				

				curr_seq = all_val_grps[grp_idx][seq_idx:seq_idx+5]
				curr_seq =[x.split('/')[-1][0:-4] for x in curr_seq]

				curr_seq_segs = [];
				curr_seq_odo = [];
				for frame in curr_seq:
					if frame in f.keys() and frame in all_val_odo_keys:
						curr_seq_segs.append(f[frame][()]);
						curr_seq_odo.append( np.expand_dims(self.val_odo_data[frame], axis=0) );
					else:
						continue;

				if len(curr_seq_segs) < 5:
					continue;


				city = curr_seq[-1].split('_')[0];
				seq_num = curr_seq[-1].split('_')[1];
				frame_num = curr_seq[-1].split('_')[2];

				gt_name = '/BS/cityscapes00/Cityscapes/gtFine/val/' + city + '/' + city + '_' + seq_num + '_' + frame_num + '_' + 'gtFine_labelTrainIds.png'

				gt = cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE);
				#print(np.unique(gt))
				gt = cv2.resize(gt, (0,0), fx=0.125, fy=0.125)

				data_X_s.append( np.concatenate(curr_seq_segs[0:4], axis = -1) )
				data_X_o.append( np.concatenate(curr_seq_odo, axis = 0) )

				data_Y_.append(curr_seq_segs[-1] )
				data_Y.append( np.expand_dims(gt,axis=2) )

				if len(data_Y) == num_of_test_examples:
					break;

			data_X_s = np.array(data_X_s);
			data_X_o = np.array(data_X_o);
			data_Y = np.array(data_Y);
			data_Y_ = np.array(data_Y_);
			f.close();

			data_X_s = np.expand_dims(data_X_s, axis=1);
			data_X_s = np.repeat(data_X_s,test_samples,axis=1)
			data_X_s = np.reshape(data_X_s,(data_X_s.shape[0]*test_samples,data_X_s.shape[2],data_X_s.shape[3],data_X_s.shape[4]));

			data_X_o = np.expand_dims(data_X_o, axis=1);
			data_X_o = np.repeat(data_X_o,test_samples,axis=1)
			data_X_o = np.reshape(data_X_o,(data_X_o.shape[0]*test_samples,data_X_o.shape[2],data_X_o.shape[3]));

			np.save('val_gt.npy',data_Y)

			return ( data_X_s, data_X_o, data_Y, data_Y_)



