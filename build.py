import model_data
def get_build():
	# Define Params
	params = {
			'learning_rate':0.001,
			'batch_size':20,
			'n_batches':2,
			'n_steps':20,
			'n_hidden':128,
			'n_input':8
			}

	# Set Data to your data class
	data = model_data.SineData(params)

	return data, params

if __name__=="__main__":
	_, params = get_build()
	print(params)
