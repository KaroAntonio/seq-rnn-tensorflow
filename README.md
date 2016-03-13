# Sequence LSTM-RNN
(in tensorflow)

This is an lstm that deals with sequences and is designed for predicting the next vector in a sequence of vectors.  

Data Form: a sequence of 1-d vectors with values in [0,1]  

The data class generates two inversed sine waves in order to test the capabilities of the rnn and investigate it's limitations.    
In order to be able to deal with n 'classes', the final layer of the output is activated by a sigmoid.  
Since all the data is sandbox data, it doesn't need to be preprocessed or anything.  
It's built to be modular so to work with new data, all you need to change is the Data class.  
Network and data params are isolated into params.py for easy play.

To train the model:

	python train.py
	
To generate a sequence (from a zero vector sequence seed by default):

	python generate.py

