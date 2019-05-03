import theano

import theano.tensor as T
import numpy as np
from cost import binary_entropy,categorical_entropy,categorical_weighted_entropy
from scorers import getTheanoSimilarityScore


class LogisticMultiLabelClassifier:
    def __init__(self,input,n_in,n_out,L1=0,L2=0):

        """ Initialize the parameters of the classifier

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type activation: a reference to a theano.tensor function
        :param activation: a function used for projecting the input into
                            the output space defaults to tensor.nnet.softmax

        :type L1: float
        :param L1: a float between 0 and 1 for L1 cost penalization

        :type L2: float
        :param L2: a float between 0 and 1 for L2 cost penalization

        """

        self.W = theano.shared(np.asarray(np.random.random((n_in,n_out)), dtype=theano.config.floatX),name='W',borrow = True)
        self.b = theano.shared(np.asarray(np.random.random((n_out,)), dtype=theano.config.floatX),name='b',borrow = True)

        pre_activation = T.dot(input,self.W)+self.b

        if pre_activation.ndim > 2:
            newshape = (T.prod(pre_activation.shape[:-1]),pre_activation.shape[-1])
            pre_activation = pre_activation.reshape(newshape)
	
        self.p_given_y = T.nnet.softmax(pre_activation)

        self.probs = self.p_given_y[:,1:]

        self.y_pred = T.argmax(self.p_given_y,axis=-1)

        self.params = (self.W,self.b)

        self.input = input

        self.L1 = L1
        self.L2 = L2
        self.n_out = n_out

    def categorical_entropy(self,y,mask=None):
        p_given_y = self.p_given_y[T.arange(T.prod(y.shape)),y.flatten()]
        return categorical_entropy(p_given_y,mask) + self.L1*T.sum(abs(self.W)) + self.L2 * T.sum(self.W**2)

    def cost(self,y,mask=None,cls=[0,1],weighted=False):
	
        if not weighted:
            return self.categorical_entropy(y,mask)
        else:               
            return self.categorical_weighted_entropy(y,cls,mask)

    def log_softmax(self,x):
        xdev = x - x.max(1, keepdims=True)
        return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

    def categorical_weighted_entropy(self,y,cls,mask=None):
        p_given_y = self.p_given_y[T.arange(T.prod(y.shape)),y.flatten()]
	return categorical_weighted_entropy(y.flatten(),p_given_y,cls,self.y_pred.flatten(),mask) + self.L1*T.sum(abs(self.W)) + self.L2 * T.sum(self.W**2)

    def error(self,y,mask=None,cls=(0,1),weighted=False):
        """ Return the current zero-one error count

        :type y: theano.tensor.TensorType of type int
        :param y: Tensor containing the expected predictions of the current batch

        """
        y_pred = self.y_pred
        y = y.flatten()

        if y.ndim != y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', y_pred.type)
            )
        # check if y is of the correct datatype

        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction

            if not weighted:
                if mask is None:
                    return T.mean(T.neq(y_pred,y))
                else:
                    return T.sum(mask.flatten()*T.neq(y_pred,y))*1.0/T.sum(mask.flatten())
            else:
                error = 0.0
                for l in cls:
                    if mask is None:
                        error += T.sum(T.eq(y,l)*T.neq(y_pred,l))*1.0/T.sum(T.eq(y,l))
                    else:
                        error += T.sum(T.eq(y,l)*mask.flatten()*T.neq(y_pred,l))*1.0/T.sum(T.eq(y,l)*mask.flatten())

                return error/len(cls)
        else:
            raise NotImplementedError()

    def fit(self,X_train,Y_train):
        """ Trains the current model
        :type X_train: a numpy ndarray
        :param X_train: the training set of same
        """
        pass

    def predict(self,X_test):
        prd = theano.function([self.input],[self.y_pred])
        return prd(X_test)

    def getSimilarityScore(self,y,cls,mask=None):
        """ Return the current dice score
        :type y: theano.tensor.TensorType of type int
        :param y: Tensor containing the expected predictions of the current batch

        :type cls: theano.tensor.TensorType of type int
        :param cls: Tensor containing the foreground class labels
        """

        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
               ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype

        if y.dtype.startswith('int'):
            return getTheanoSimilarityScore(self.y_pred,y,cls,mask)
        else:
            raise NotImplementedError()

