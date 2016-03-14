import math
import numpy as np
from data_utils.data import Data


def build():
	# Define Params
	params = {
			'learning_rate':0.001,
			'batch_size':15,
			'n_steps':10,
			'n_hidden':128,
			'train_steps':10000,
			'display_step':50,
			'save_step':100,
			'gen_steps':50,
			'model_ckpt':None
			}

	# Set Data to your data class
	data = SineData(params)

	return data, params

class SineData(Data):
	def __init__(self, params):
		Data.__init__(self,params)

	def post_process(self, sequence):
		# sequence
		# postprocess a sequence in some way

		# ex. convert each vec to binary and print
		for vec in sequence:
			n_hot = [int(e+0.5) for e in vec]
			ones = str(n_hot).replace('0',' ')
			print(ones)

	def get_seed(self):
		return self.zero_batch()

	def load_data(self):
		# seq data is one giant sequence of data,
		# from which x and y are taken.
		self.n_batches = self.params['n_batches'] = 2
		self.n_input = self.params['n_input'] = 10

		wave = self.gen_wave_1()
		return wave, wave

	def gen_wave_2(self):
		# Vector to vector sine wav
		data = []
		num_datas = (self.n_batches*
					self.batch_size*
					self.n_steps*
					(self.n_input/2))
		for i in range(num_datas):
			wave = (math.sin(i/5.)+1)/2.
			data += [[wave] * self.n_input ]

		return np.array(data)

	def gen_wave_1(self):
			# Double intersecting sine wave
			data = []
			num_datas = (self.n_batches*
						self.batch_size*
						self.n_steps*
						(self.n_input/2))
			for i in range(num_datas):
				wave = int((math.sin(i/5.)+1)*self.n_input/2)
				hot_wave = np.array(self.to_one_hot(wave, self.n_input))
				data += [hot_wave[::-1] + hot_wave]

			return np.array(data)


if __name__ == "__main__":
    data,params = build()
    print(params)
    x,y = data.next_train()
