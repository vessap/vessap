import theano
import theano.tensor as T
import numpy as np
from theano.tensor import nnet as conv
from theano.tensor.nnet import conv3d
import theano.tensor.nnet as nnet
from theano.tensor.signal import pool as downsample
from theano.tensor.shared_randomstreams import RandomStreams

class Conv2dLayer:
    def __init__(self,input,ker_shape=(5,1,3,3),pool_size=(2, 2),w=None,b=None,activation=None,border_mode='valid',rng = np.random.RandomState(23125),divisions=1):

        """

            :type input: theano.tensor4
        	:param input: theano.tensor4 symbolic variable

        	:type ker_shape: tuple of length 4
        	:param ker_shape: tuple of size (output_maps,input_maps,neigbourhood_x,neighbourhood_y) default to (1,5,3,3)

        	:type pool_size: tuple
        	:param pool_size: tuple of size (pool_x,pool_y) default to (2,2)

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
        w_shp = fil_shape

        w_bound = np.sqrt(fil_shape[1]*fil_shape[2]*fil_shape[3])

        self.input = input

        if w is None:
            w = theano.shared(np.asarray(
                rng.uniform(
                    low=-1.0 / w_bound,
                    high=1.0 / w_bound,
                    size=w_shp),
                dtype=self.input.dtype), name ='W',borrow=True)

        if b is None:
            b_shp = (fil_shape[0],)
            b = theano.shared(np.asarray(
                rng.uniform(low=-.5, high=.5, size=b_shp),
                dtype=self.input.dtype), name='b',borrow=True)

        if activation is None:
            activation = T.tanh

        self.W = w
        self.b = b

        padded_out = self.input
        act_b_mode = border_mode

        if border_mode=='same':
            ns = (self.input.shape[0],self.input.shape[1],self.input.shape[2]+(ker_shape[2]-1),self.input.shape[3]+(ker_shape[3]-1))
            padded_out = T.zeros(shape=ns,dtype=self.input.dtype)
            padded_out = T.set_subtensor(padded_out[:,:,(ker_shape[2]-1)/2:-(ker_shape[2]-1)/2,(ker_shape[3]-1)/2:-(ker_shape[3]-1)/2],self.input)
            act_b_mode = 'valid'


        if divisions > 1:
            for i in range(divisions):
                wid = padded_out.shape[0]/divisions
                if i==0:
                    conv_out = conv.conv2d(padded_out[i*wid:(i+1)*wid],self.W,border_mode=act_b_mode)
                    if not (pool_size[0]==1 and pool_size[1]==1):
                        pool_out = downsample.pool_2d(input=conv_out,ds=pool_size,ignore_border=True)
                    else:
                        pool_out = conv_out
                else:
                    if i==(divisions-1):
                        conv_out = conv.conv2d(padded_out[i*wid:],self.W,border_mode=act_b_mode)

                        if not (pool_size[0]==1 and pool_size[1]==1):
                            temp_pool_out = downsample.pool_2d(input=conv_out,ds=pool_size,ignore_border=True)
                            pool_out = T.concatenate([pool_out,temp_pool_out])
                        else:
                            temp_pool_out = conv_out
                            pool_out = T.concatenate([pool_out,temp_pool_out])

                    else:
                        conv_out = conv.conv2d(padded_out[i*wid:(i+1)*wid],self.W,border_mode=act_b_mode)
                        if not (pool_size[0]==1 and pool_size[1]==1):
                            temp_pool_out = downsample.pool_2d(input=conv_out,ds=pool_size,ignore_border=True)
                            pool_out = T.concatenate([pool_out,temp_pool_out])
                        else:
                            temp_pool_out = conv_out
                            pool_out = T.concatenate([pool_out,temp_pool_out])

        else:
            conv_out = conv.conv2d(padded_out,self.W,border_mode=act_b_mode)
            if not (pool_size[0]==1 and pool_size[1]==1):
                pool_out = downsample.pool_2d(input=conv_out,ws=pool_size,ignore_border=True)
            else:
                pool_out = conv_out



        self.output = activation(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = (self.W,self.b)
        self.ker_shape = fil_shape
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

class Conv_2d_to_3d:
    def __init__(self,input,ker_size=(3,3,3),bank_size=1,input_maps=1,output_maps=1,activation=T.tanh):
        self.params = ()
        self.input = input

        x_s = self.input.shape
        n_x,n_y,n_z = ker_size

        x_params = ()
        y_params = ()
        z_params = ()


        for i in range(bank_size):
            x_w = self.make_params((output_maps,input_maps,n_y,n_z))

            conv2d_xi = nnet.conv3d(input=self.input,filters=x_w.reshape((output_maps,input_maps,1,n_y,n_z)),border_mode='half')
            x_params = x_params + (x_w,)

            y_w = self.make_params((output_maps,input_maps,n_x,n_z))

            conv2d_yi = nnet.conv3d(input=self.input,filters=y_w.reshape((output_maps,input_maps,n_x,1,n_z)),border_mode='half')
            y_params = y_params + (y_w,)

            z_w = self.make_params((output_maps,input_maps,n_x,n_y))

            conv2d_zi = nnet.conv3d(input=self.input,filters=z_w.reshape((output_maps,input_maps,n_x,n_y,1)),border_mode='half')
            z_params = z_params + (z_w,)

            if(i==0):
                self.output = (conv2d_xi + conv2d_yi + conv2d_zi)/3.0
            else:
                self.output = self.output + (conv2d_xi + conv2d_yi + conv2d_zi)/3.0

        w,b = self.make_params((output_maps,input_maps,n_x,n_y,n_z),True)
        self.output = self.output * 1.0 / bank_size
        self.output = activation(self.output + b.dimshuffle('x',0,'x','x','x'))
        self.params = x_params + y_params + z_params + (b,)

    def make_params(self, ker_shape,with_b=False, w=None, b=None,rng=np.random.RandomState(1235)):
        fil_shape = ker_shape
        w_shp = fil_shape

        w_bound = np.sqrt(fil_shape[1]*fil_shape[2]*fil_shape[3])

        if w is None:
            w = theano.shared(np.asarray(
                rng.uniform(
                    low=-1.0 / w_bound,
                    high=1.0 / w_bound,
                    size=w_shp),
                dtype=self.input.dtype), name ='W',borrow=True)

        if b is None and with_b:
            b_shp = (fil_shape[0],)
            b = theano.shared(np.asarray(
                rng.uniform(low=-.5, high=.5, size=b_shp),
                dtype=self.input.dtype), name='b',borrow=True)

        if with_b:
            return w,b
        else:
             return w

    def set_params(self,params):
        """ Set the values of the parameters for this layer

        	:type params: list or tuple of numpy.ndarrays
        	:param params: a list or tuple of save numpy.ndarray. NOTE the length and shapes should match the shape and length of the layer
		"""
        if len(self.params) != len(params):
            raise TypeError('Parameters size mismatch!','layer has %i parameters but supplied %i parameters'%(len(self.params),len(params)))

        for i in range(len(self.params)):
            try:
                self.params[i].set_value(params[i])
            except:
                raise TypeError('Parameter shape mismatch','layer parameter %i of and supplied values do not match'%(i,))

    def get_output(self):
        """ Returns a symbolic output of this layer which can be used in a multi - layer network

		"""
        return self.output
