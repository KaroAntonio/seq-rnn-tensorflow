import numpy as np
import math
from params import *

class Data:
	def __init__(self, params):
		self.params = params
		self.train_pointer = -1
		self.test_pointer = -1

		self.batch_size = params['batch_size']
		self.n_input = params['n_input']
		self.seq_len = params['n_steps']
		self.n_batches = params['n_batches']	
	
		self.prep_data()


	def prep_data(self):
		# Create Dats
		# data is a set of sequences of shape N x batch_size
		self.train = self.create_data()
		self.test = self.create_data()

		# Prep Btches
		self.train_x, self.train_y = self.prep_batches(self.train)
		self.test_x, self.test_y = self.prep_batches(self.test)
	
	def zero_batch(self):
		# Return a batch of zeros
		x_shape = ((self.batch_size,self.seq_len,self.n_input))
		y_shape = ((self.batch_size,self.n_input))
		x = np.zeros(x_shape)
		y = np.zeros(y_shape)
		return x,y

	def prep_batches(self, data):
		x_data = []
		y_data = []
		for i in range(self.seq_len,len(data)):
			x_data += [data[i-self.seq_len:i]]
			y_data += [data[i]]

		self.num_batches = len(x_data)//self.batch_size
		x_data = x_data[:self.num_batches * self.batch_size]
		y_data = y_data[:self.num_batches * self.batch_size]

		x = np.split(np.array(x_data),self.num_batches)
		y = np.split(np.array(y_data),self.num_batches)
		return x,y

	def decode_one_hot_seq(self, one_hot_seq):
		# Decodes a vector of onehots to a vector of indices
		return [oh.index(1) for oh in one_hot_seq]

	def post_process(self, sequence):
		# sequence 
		# postprocess a sequence in some way

		# ex convert each vec to binary and print
		for vec in sequence:
			n_hot = [int(e+0.5) for e in vec]
			print(n_hot)

	def create_data(self):
		# seq data is one giant sequence of data, from which x and y are taken.
		return self.gen_wave_1()
		
	def gen_wave_2(self):
		# Vector to vector sine wav
		data = []
		self.sample_dim = self.n_input
		self.params['n_input'] = self.sample_dim
		num_datas = self.n_batches*self.batch_size*self.seq_len*(self.n_input/2)
		for i in range(num_datas):
			wave = (math.sin(i/5.)+1)/2.
			data += [[wave] * self.sample_dim ]

		return np.array(data)

	def gen_wave_1(self):
		# Double intersecting sine wave
		data = []
		self.sample_dim = self.n_input
		self.params['n_input'] = self.sample_dim
		num_datas = self.n_batches*self.batch_size*self.seq_len*(self.n_input/2)
		for i in range(num_datas):
			wave = int((math.sin(i/5.)+1)*self.n_input/2)
			hot_wave = np.array(self.to_one_hot(wave,self.sample_dim))
			data += [hot_wave[::-1] + hot_wave]

		return np.array(data)


	def to_one_hot(self, i, size):
		one_hot = np.zeros(size)
		one_hot[i] = 1
		return np.array(one_hot)

	def hot_to_i(self, one_hot):
		return one_hot.index(1)
	
	def next_batch(self, x_batches,y_batches,ptr):
		return x_batches[ptr],y_batches[ptr]

	def next_train(self):
		#return x,y for next train batch
		self.train_pointer = (self.train_pointer+1)%self.num_batches
		return self.next_batch(self.train_x,self.train_y,self.train_pointer)

	def next_test(self):
		#return x,y for next test batch
		self.test_pointer = (self.test_pointer+1)%self.num_batches
		return self.next_batch(self.test_x,self.test_y,self.test_pointer)

class SineData(Data):
	def __init__(params):
		Data.__init__(params)
   
if __name__ == "__main__":
	params = get_params()
	data = Data(params)
	x,y = data.next_train()
	print(x)
	print(y)
