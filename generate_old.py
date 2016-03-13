
from seq_data_util import SeqData
from model import RNN

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

'''
LSTM RNN For Predicting the Next element in a sequence (in one hot)
'''

# Slice Classification

# Parameters
learning_rate = 0.001
training_iters = 12000
batch_size = 20
display_step = 10

# Network Parameters
n_batches = 2	
n_steps = 20 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 4  # total classes (0-9 digits)
one_hot = True

# Generation Parameters
gen_seq_len = 100

print("Creating Data...")
data = SeqData(batch_size,n_classes,n_steps, n_batches, one_hot)
n_input = data.sample_dim

# tf Graph input
# dims of input will be a batch of known size, with strings of unknown length, with one-hot chars of known length
x = tf.placeholder("float", [batch_size, n_steps, n_input])
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate = tf.placeholder("float", [None, 2*n_hidden])
y = tf.placeholder("float", [None, n_input])

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_input]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_input]))
}

print("Building Model...")
params =  {
    'n_steps':n_steps,
    'n_hidden':n_hidden,
    'n_input':n_input
}

pred = RNN(x, istate, weights, biases,params)

# Sigmoid output activation is chosen in order to choose n buckets
pred_ = tf.nn.sigmoid(pred)


# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y)) # Sigmoid loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model 
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
# Cosine Similiarity
dot = tf.reduce_sum(tf.mul(y,pred_))
y_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(y))))
pred_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.abs(pred_))))
norms = tf.mul(y_norm,pred_norm)
cosine_similiarity = tf.div(dot, norms)

accuracy = tf.reduce_mean(tf.cast(cosine_similiarity, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Saver
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
	# sess.run(init)
	saver.restore(sess, "save/model.ckpt")
	
	# A Batch is a bunch of sequences
	# TODO fix to accept one sequence at a time
	x_feed,_ = data.zero_batch()

	gen_seq = x_feed[0].tolist()

	for i in range(gen_seq_len):


		feed_dict = {
			    x: x_feed,
			    y: _,
			    istate: np.zeros((batch_size, 2*n_hidden))
			    }

		_next = sess.run(pred_, feed_dict=feed_dict)
		n_hot = [int(e+0.5) for e in _next[0]]

		print (n_hot)
		
		gen_seq.append(_next[0].tolist())
		#set the feed for the next round of sampling
		# grab last seq_len samples off gen_seq
		next_feed = np.array(gen_seq[-data.seq_len:])
		x_feed[0] = np.array(gen_seq[-data.seq_len:])

	

