import theano
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

class Fully_Connected_Layer(object):
    def __init__(self,input,n_in,n_out,W=None,b=None,activation=None,rng=np.random.RandomState(1235)):
        """ Initialize and build the graph of the hiddenlayer

        	:type input: theano.tensor.TensorType
        	:param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        	:type n_in: int
        	:param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        	:type n_out: int
        	:param n_out: number of output units, the dimension of the space in
                      which the labels lie

        	:type W: theano.tensor.TensorType
        	:param W: symbolic variable that describes
                  Weight for this Hidden Layer. defualt to a random sample
                  in the interval Wij < |sqrt(6. / (n_in + n_out))|

        	:type b: theano.tensor.TensorType
        	:param b: symbolic variable that describes
                  bais for this Hidden Layer. defualt to a vetor of zeros

        	:type activation: a reference to a theano.tensor function
        	:param activation: a function used for projecting the input into
                            the output space defaults to tensor.tanh

            	:type rng: numpy.random.RandomState
        	:param rng: a random number generator used to initialize weights

		    """

        if activation is None:
            activation = T.tanh

        if W is None:
            W_values = np.asarray(rng.uniform(
                    				low=-np.sqrt(6. / (n_in + n_out)),
                    				high=np.sqrt(6. / (n_in + n_out)),
                    				size=(n_in, n_out)
                				),
                		dtype=theano.config.floatX
            			)

            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

        self.W = theano.shared(value=W_values, name='W', borrow=True)


        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)

        self.b = theano.shared(value=b_values, name='b', borrow=True)

        self.input = input
        self.output = activation(T.dot(self.input,self.W)+self.b)
        self.params = (self.W,self.b)
        self.activation = activation
        self.n_in = n_in
        self.n_out = n_out


    def set_params(self,W=None,b=None):
        """ Set the values (W,b) of the parameters for this layer

        	:type W: numpy.ndarray
        	:param W: numpy array of size (n_in,n_out)

        	:type b: numpy.ndarray
        	:param b: numpy vector of size (n_out,)

		    """
        if not W is None:
            self.params[0].set_value(W)

        if not b is None:
            self.params[1].set_value(b)

    def get_output(self):
        """ Returns a symbolic output of this layer which can be used in a multi - layer network

		    """
        return self.output
