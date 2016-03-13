import numpy as np
import math
from params import *

class SeqData:
	def __init__(self, params):
		self.params = params
		self.one_hot = True	
		self.batch_size = params['batch_size']
		self.n_classes = params['n_input']/2
		self.seq_len = params['n_steps']
		self.n_batches = params['n_batches']	
		self.n_size = self.n_classes + self.seq_len - 1 if self.one_hot else 1
		# pointers in arrays tp be able to pass by reference
		self.train_pointer = -1
		self.test_pointer = -1
		
		self.prep_data()


	def prep_data(self):
		# Create Dats
		#data is a set of sequences of len N x batch_size
		self.train = self.create_data()
		self.test = self.create_data()

		# Prep Btches
		self.train_x, self.train_y = self.prep_batches(self.train)
		self.test_x, self.test_y = self.prep_batches(self.test)
	
	def zero_batch(self):
		# Return a batch of zeros
		x_shape = ((self.batch_size,self.seq_len,self.n_classes*2))
		y_shape = ((self.batch_size,self.n_classes*2))
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

	def decodeOneHotSeq(self, one_hot_seq):
		# Decodes a vector of onehots to a vector of indices
		return [oh.index(1) for oh in one_hot_seq]


	def create_data(self):
		#seq data is one giant sequence of data, from which x and y are taken.
		return self.gen_wave_1()
		
	def gen_wave_2(self):
		data = []
		tokens = [i for i in range(1,self.n_classes)]
		#self.sample_dim = sum(tokens) #the dimensions of each sample
		self.sample_dim = self.n_classes*2
		self.params['n_input'] = self.sample_dim
		for i in range(self.n_batches*self.batch_size*self.seq_len*self.n_classes):
			wave = (math.sin(i/5.)+1)/2.
			data += [[wave] * self.sample_dim ]

		return np.array(data)

	def gen_wave_1(self):
		data = []
		tokens = [i for i in range(1,self.n_classes)]
		#self.sample_dim = sum(tokens) #the dimensions of each sample
		self.sample_dim = self.n_classes*2
		self.params['n_input'] = self.sample_dim
		for i in range(self.n_batches*self.batch_size*self.seq_len*self.n_classes):
			wave = int((math.sin(i/5.)+1)*self.n_classes)
			hot_wave = np.array(self.toOneHot(wave,self.sample_dim))
			bass = np.array(self.toOneHot(2,self.sample_dim))
			data += [hot_wave[::-1] + hot_wave]

		return np.array(data)


	def toOneHot(self, i, size):
		one_hot = np.zeros(size)
		one_hot[i] = 1
		return np.array(one_hot)

	def hotToI(self, one_hot, size):
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

   
if __name__ == "__main__":
	params = get_params()
	data = SeqData(params)
	x,y = data.next_train()
	print(x)
	print(y)
