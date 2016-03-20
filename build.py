import math
import numpy as np
from data_utils.data import Data
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def build():
	# Define Params
	params = {
			'ckpt':None,
			'learning_rate':0.001,
			'batch_size':20,
			'n_steps':30,
			'n_hidden':128,
			'train_steps':60000,
			'display_step':50,
			'save_step':100,
			'gen_steps':300,
			}

	# Set Data to your data class
	return SineData(params), params

class SineData(Data):
	def __init__(self, params):
		Data.__init__(self,params)

	
	def post_process(self, sequence):
		# postprocess a sequence in some way

		sequence = np.array(sequence[self.params['n_steps']:])
		mpimg.imsave('sequence.png',sequence)

		# ex. convert each vec to binary and print
		fig = plt.figure(figsize=(10,10))
		sub = fig.add_subplot(111)
		sub.matshow(sequence,  cmap=plt.cm.gray)

		plt.show()

	def get_seed(self):
		return self.next_train()

	def load_data(self):
		'''
		Return x, y sequences from which batches are sampled
		'''
		self.n_input = self.params['n_input'] = 200
		
		num_samples = (100*self.n_input)

		wave = self.gen_wave_3(num_samples)
		return wave, wave

	def gen_wave_3(self,n):
		# Vector to vector sine wav
		data = []
		for i in range(n):
			wave_a = (math.sin(i/5.)+1)/6.
			wave_b = np.array([(math.sin((j+i)/10.)+1)/6. for j in range(self.n_input)])
			wave_c = np.array([(math.sin((i-j)/5.)+1)/6. for j in range(self.n_input)])
			rand = (np.random.random(self.n_input)/10)
			data += [rand+wave_a+wave_b+wave_c]

		return np.array(data)


	def gen_wave_2(self,n):
		# Vector to vector sine wav
		data = []
		for i in range(n):
			wave = (math.sin(i/10.)+1)/2.
			data += [(np.random.random(self.n_input)/10+wave)]

		return np.array(data)

	def gen_wave_1(self,n):
		# Double intersecting sine wave
		data = []
		for i in range(n):
			wave = int((math.sin(i/5.)+1)*self.n_input/2)
			hot_wave = np.array(self.to_one_hot(wave, self.n_input))
			data += [hot_wave[::-1] + hot_wave]

		return np.array(data)

	def show(self):
		#seq,_ = self.load_data()
		seq,_ = self.next_train()
		seq = seq.reshape(seq.shape[0]*seq.shape[1],seq.shape[2])

		fig = plt.figure(figsize=(10,10))
		sub = fig.add_subplot(111)
		sub.matshow(seq[:500],  cmap=plt.cm.gray)
		plt.show()

if __name__ == "__main__":
	data,params = build()
	print(params)
	x,y = data.next_train()
	data.show()
