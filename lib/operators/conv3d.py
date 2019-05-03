from theano.tensor import TensorType
import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet import conv
from theano.tensor.nnet import conv3d
from theano.tensor.signal import pool as downsample
from theano.tensor.shared_randomstreams import RandomStreams
from .conv2d import Conv_2d_to_3d
from theano.tensor.nnet.neighbours import images2neibs

tensor5 = TensorType(theano.config.floatX, (False,)*5)

class Conv3dLayer:
    def __init__(self,input,ker_shape=(5,1,3,3,3),pool_size=(2,2,2),w=None,b=None,activation=None,border_mode='valid',rng = np.random.RandomState(23125)):
        """

            :type input: theano.tensor5
        	:param input: theano.tensor5 symbolic variable

        	:type ker_shape: tuple of length 5
        	:param ker_shape: tuple of size (output_maps,input_maps,neigbourhood_x,neighbourhood_y,neighbourhood_z) default to (1,5,3,3,3)

        	:type pool_size: tuple
        	:param pool_size: tuple of size (pool_x,pool_y,pool_z) default to (2,2,2)

        	:type W: numpy.ndarray
        	:param W: numpy array of size (n_in,n_out) default to None

        	:type b: numpy.ndarray
        	:param b: numpy vector of size (n_out,) default to None

        	:type activation: theano symbolic function
        	:param activation: nonlinear activation function Default to theano.tensor.tanh

        	:type rng: numpy.random.RandomState
        	:param rng: a random state object for generating random numbers for parameters

		"""

        fil_shape = ker_shape
        w_shp = ker_shape

        w_bound = np.power(np.product(ker_shape[1:]),1.0/3)

        self.input = input

        if w is None:
            w = theano.shared(np.asarray(
                rng.uniform(
                    low=-1.0 / w_bound,
                    high=1.0 / w_bound,
                    size=w_shp),
                dtype=self.input.dtype), name='W',borrow=True)

        if b is None:
            b_shp = (fil_shape[0],)
            b = theano.shared(np.asarray(
                rng.uniform(low=-.5, high=.5, size=b_shp),
                dtype=self.input.dtype), name='b',borrow=True)

        if activation is None:
            activation = T.tanh

        self.W = w
        self.b = b

        if border_mode=='same':
	    border_mode = 'half'

        conv_out = conv3d(input=self.input,filters=self.W,border_mode=border_mode)

        if pool_size != (1,1,1):
            pool_out = max_pool_3d(input=conv_out,ws=pool_size,ignore_border=True)
            self.output = activation(pool_out + b.dimshuffle('x',0,'x','x','x'))
        else:
            self.output = activation(conv_out + b.dimshuffle('x',0,'x','x','x'))

        self.params = (self.W,self.b)
        self.ker_shape = ker_shape
        self.pool_size = pool_size
        self.activation = activation
        self.border_mode = border_mode

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
