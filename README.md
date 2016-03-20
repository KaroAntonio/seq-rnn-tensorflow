# Sequence LSTM-RNN
(in tensorflow)

This is an lstm that deals with sequences and is designed for predicting the next vector in a sequence of vectors.  

It's built to be modular so to work with new data, all you need to change is the build file.  

Data Form: a sequence of 1-d vectors with values in [0,1]  

The data class generates two inversed sine waves in order to test the capabilities of the rnn and investigate it's limitations.    
In order to be able to deal with n 'classes', the final layer of the output is activated by a sigmoid.  
Since all the data is sandbox data, it doesn't need to be preprocessed or anything.  
It's built to be modular so to work with new data, all you need to change is the build file.  
Network and data params are defined in the data build function.

#### Dependencies

[Numpy](http://www.scipy.org/scipylib/download.html)  
[Tensorflow](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html)

#### Train

	python train.py
	
#### Generate

	python generate.py

