from nn_utils.model import Model
from build import build

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

'''
LSTM RNN For Generating the Next Vector Element in a Sequence 
'''
# Build Data, Params
data, params = build()

# Parameters
gen_seq_len = params['gen_steps']

print('Building Model...')
model = Model(params)

# Initializing the variables
init = tf.initialize_all_variables()

# Saver
saver = tf.train.Saver()

def istate():
	return np.zeros((params['batch_size'], 2*params['n_hidden']))

# Launch the graph
with tf.Session() as sess:
	saver.restore(sess, "save/model.ckpt")

    # TODO fix to accept one sequence at a time
	x_feed,_ = data.get_seed()
	gen_seq = x_feed[0].tolist()
	for i in range(gen_seq_len):

		feed_dict = {
				model.x: x_feed,
				model.y: _,
				model.initial: istate() 
				}
	
		# Predict
		_next = sess.run(model.pred, feed_dict=feed_dict)

		# Append prediction to generated sequence
		gen_seq.append(_next[0].tolist())
		
		# Update Feed
		next_feed = np.array(gen_seq[-params['n_steps']:])
		x_feed[0] = next_feed 

data.post_process(gen_seq)	

		
