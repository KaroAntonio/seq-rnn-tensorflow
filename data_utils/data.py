import numpy as np
import math
import random

class Data:
	def __init__(self, params):
		self.params = params
		self.train_pointer = -1
		self.test_pointer = -1

		self.batch_size = params['batch_size']
		self.n_steps = params['n_steps']

		self.prep_data()
		
	def load_data(self):
		'''
		Return 2 numpy array of 1-d vectors 
			of shape (-, n_input)
			with values in range [0,1]
			for training and testing
		return train, test
		'''
		pass

	def post_process(self, sequence):
		'''
		Process sequence
		'''
		pass

	def get_seed(self):
		'''
		Return seed batch
		'''

	def show(self):
		'''
		Graph or otherwise represent Data in a way that could not be done in a string repr
		'''

	def prep_data(self):
		# Create Dats
		print('Loading Data...')
		self.train, self.test = self.load_data()

		# Validate Sequence
		min_size = self.params['n_steps']*self.params['batch_size']
		if len(self.train) < min_size or len(self.test) < min_size:
			err_str = 'Data Sequence must be of length at least n_steps*batch_size: '
			raise Exception(err_str + str(min_size))
			
		self.n_input = self.params['n_input'] = self.train.shape[1]
		self.params['n_batches'] = self.train.shape[0]//self.batch_size
		self.n_batches = self.params['n_batches']

		# Prep Data - Double the train data to easily deal with batch wrap-around
		self.train_x = np.array(list(self.train)*2)
		self.train_y = np.array(list(self.train)*2)
		self.test_x = np.array(list(self.test)*2)
		self.test_y = np.array(list(self.test)*2)

	def decode_one_hot_seq(self, one_hot_seq):
		# Decodes a vector of onehots to a vector of indices
		return [oh.index(1) for oh in one_hot_seq]

	def zero_batch(self):
		# Return a batch of zeros
		x_shape = ((self.batch_size, self.n_steps, self.n_input))
		y_shape = ((self.batch_size, self.n_input))
		x = np.zeros(x_shape)
		y = np.zeros(y_shape)
		return x,y

	def to_one_hot(self, i, size):
		'''
		Return a one hot rep of i in a vec size=size
		'''
		one_hot = np.zeros(size)
		one_hot[i] = 1
		return np.array(one_hot)

	def hot_to_i(self, one_hot):
		'''
		Return the int version of one_hot
		'''
		return one_hot.index(1)

	def rand_batch(self, x_data, y_data):
		'''
		Return batch of sequences randomly sampled from x,y data
		'''
		x_batch = []
		y_batch = []
		rands = np.random.random(self.batch_size)
		for r in rands:
			i = int(r*(len(x_data)-self.n_steps-1))
			x_batch += [x_data[i:i+self.n_steps]]
			y_batch += [y_data[i+self.n_steps]]
		return np.array(x_batch), np.array(y_batch)


	def next_batch(self, x_data, y_data, ptr):
		'''
		Return the next batch pointed to by ptr
		'''
		return self.rand_batch(x_data,y_data)

	def next_train(self):
		#return x,y for next train batch
		self.train_pointer = (self.train_pointer+1)%self.n_batches
		return self.next_batch(self.train_x, self.train_y, self.train_pointer)

	def next_test(self):
		#return x,y for next test batch
		self.test_pointer = (self.test_pointer+1)%self.n_batches
		return self.next_batch(self.test_x, self.test_y, self.test_pointer)

	def __len__(self):
		return len(self.train)

