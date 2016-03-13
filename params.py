def get_params():
	return {
			'learning_rate':0.001,
			'batch_size':30,
			'n_batches':30,
			'n_steps':10,
			'n_hidden':128,
			'n_samples':30,
			'n_input':8
			}

if __name__=="__main__":
	params = get_params()
	print(params)
