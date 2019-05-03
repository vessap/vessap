from __future__ import print_function
import theano
from theano.tensor import TensorType
import theano.tensor as T
import numpy as np
from ..operators import LogisticMultiLabelClassifier,Conv_2d_to_3d
from ..utilities import load_data,save_data
itensor5 = T.itensor5
tensor5 = T.tensor5

class FCNN(object):
    def __init__(self,n_out,conv_layers):
        self.n_out = n_out
        self.conv_layers = conv_layers
        self.build_model_3d()
        self.input_params = (n_out,conv_layers)

    def build_model_3d(self):
        self.input = T.tensor5('input')
        self.params = ()

        c_input = self.input
        feat_maps = 1
        for ker,actfunc in self.conv_layers:
            conv = Conv_2d_to_3d(input=c_input,ker_size=ker[-3:],bank_size=1,input_maps=ker[1],output_maps=ker[0],activation=actfunc)
            c_input = conv.output
            self.params += conv.params
            feat_maps = ker[0]

        c_h_input = T.transpose(c_input,axes=(0,2,3,4,1))
        c_n_in = feat_maps

        self.classifier = LogisticMultiLabelClassifier(T.reshape(c_h_input,(T.prod(c_h_input.shape[:4]),c_h_input.shape[4])),c_n_in,self.n_out)
        self.params += self.classifier.params

    def fit(self,X,Y,n_epochs=100,batch_size=1,learning_rate=0.01,mask=None,classes=[1,],weighted_cost=False,weighted_error=False):
        self.y = T.itensor5('y')

        if mask is None:
            self.mask = None
        else:
            self.mask = itensor5('mask')

        l_rate = T.scalar('l_r')
        index = T.iscalar('index')

        if len(X.shape)==4:
            X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2],X.shape[3])


        data_set_x = theano.shared(np.asarray(X,dtype=theano.config.floatX),borrow=True)
        data_set_y = theano.shared(np.asarray(Y,dtype='int32'),borrow=True)

        if not (mask is None):
            data_set_mask = theano.shared(np.asarray(mask,dtype='int32'),borrow=True)

        cost,error = self.classifier.cost(self.y,self.mask,weighted=weighted_cost),self.classifier.error(self.y,self.mask,weighted=weighted_error)
        dice,recall,prec = self.classifier.getSimilarityScore(self.y.flatten(),classes,self.mask)

        updates = [
            (param,param-l_rate*gparam) for param,gparam in zip(self.params,T.grad(cost,self.params))
        ]

        if mask is None:
            train_model = theano.function(
                            inputs = [index,l_rate],
                            outputs = [cost,error],
                            updates = updates,
                            givens = {
                                    self.input : data_set_x[index*batch_size:(1+index)*batch_size],
                                    self.y : data_set_y[index*batch_size:(1+index)*batch_size]
                                }

                            )
            train_model_det = theano.function(
                            inputs = [index,l_rate],
                            outputs = [cost,error,dice,recall,prec],
                            updates = updates,
                            givens = {
                                    self.input : data_set_x[index*batch_size:(1+index)*batch_size],
                                    self.y : data_set_y[index*batch_size:(1+index)*batch_size]

                                }

                            )
        else:
            train_model = theano.function(
                            inputs = [index,l_rate],
                            outputs = [cost,error],
                            updates = updates,
                            givens = {
                                    self.input : data_set_x[index*batch_size:(1+index)*batch_size],
                                    self.y : data_set_y[index*batch_size:(1+index)*batch_size],
                                    self.mask : data_set_mask[index*batch_size:(1+index)*batch_size]
                                }

                            )
            train_model_det = theano.function(
                            inputs = [index,l_rate],
                            outputs = [cost,error,dice,recall,prec],
                            updates = updates,
                            givens = {
                                    self.input : data_set_x[index*batch_size:(1+index)*batch_size],
                                    self.y : data_set_y[index*batch_size:(1+index)*batch_size],
                                    self.mask : data_set_mask[index*batch_size:(1+index)*batch_size]

                                }

                            )

        n_batches = X.shape[0]/batch_size

        print("model training started........................................")
        patience = 0
        prev_cst = 9999999
        metrics = [0,0.0,0.0,0.0,0.0,0.0]
        for i in range(n_epochs):
            if i%1==0:
                cst,err,dic,rec,pre = (0.0,0.0,0.0,0.0,0.0)
                for j in range(n_batches):
                    c,e,d,r,p = train_model_det(j,learning_rate)
                    cst += c
                    err += e
                    dic += d
                    rec += r
                    pre += p

                metrics = np.append(metrics,[i+1,cst*0.1/n_batches,err*100.0/n_batches,dic*100.0/n_batches,rec*100.0/n_batches,pre*100.0/n_batches])

                print('Iteration %i \t cost : %f \t error :%f %% \t dice :%f %% \t recall :%f %% \t precision :%f %% ' % (i+1,cst*0.1/n_batches,err*100.0/n_batches,dic*100.0/n_batches,rec*100.0/n_batches,pre*100.0/n_batches))


            else:
                cst,err = (0.0,0.0)

                for j in range(n_batches):
                    c,e = train_model(j,learning_rate)
                    cst += c
                    err += e

                print('Iteration %i \t cost : %f \t error :%f %%' % (i+1,cst*0.1/n_batches,err*100.0/n_batches))
            if (cst*0.1/n_batches) > prev_cst:

                patience = patience + 1

            if patience >= 5:
                patience = 0
                learning_rate *= 0.98

            prev_cst = cst*0.1/n_batches
        print("training of network completed!")
        return learning_rate

    def predict(self,X,Y=None,mask=None,batch_size=1):
        import timeit
        index = T.iscalar('index')
        self.y = itensor5('y')

        actshape = X.shape

        if len(X.shape)==4:
            X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2],X.shape[3])

        data_set_x = theano.shared(np.asarray(X,dtype=theano.config.floatX),borrow=True)

        if Y is None:
            pred_model = theano.function(
                            inputs=[index],
                            outputs = self.classifier.y_pred,
                            givens = {
                                self.input : data_set_x[index*batch_size:(index+1)*batch_size]
                            }
            )

            n_batches = X.shape[0]/batch_size
            elapsed_times = []
            for i in range(n_batches):
                if i == 0:
                    start_time = timeit.default_timer()
                    output = pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                else:
                    start_time = timeit.default_timer()
                    tem_out =  pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                    output = np.concatenate([output,tem_out])

            return output.reshape(actshape),elapsed_times

        else:
            data_set_y = theano.shared(np.asarray(Y,dtype='int32'),borrow=True)

            pred_model = theano.function(
                            inputs=[index],
                            outputs=[self.classifier.y_pred,self.classifier.cost(self.y),self.classifier.error(self.y)],
                            givens={
                                self.input: data_set_x[index*batch_size:(index+1)*batch_size],
                                self.y: data_set_y[index*batch_size:(1+index)*batch_size]
                            }
            )

            n_batches = X.shape[0]/batch_size
            cost,error = (0.0,0.0)
            elapsed_times = []
            for i in range(n_batches):
                if i==0:
                    start_time = timeit.default_timer()
                    output,cost,error = pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                else:
                    start_time = timeit.default_timer()
                    out,cst,err = pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                    output = np.concatenate([output,out])
                    cost += cst
                    error += err

            return output.reshape(actshape),cost/n_batches,error/n_batches,elapsed_times

    def predict_probs(self,X,Y=None,mask=None,batch_size=1):
        import timeit
        index = T.iscalar('index')
        self.y = itensor5('y')

        actshape = X.shape if X.ndim == 4 else (X.shape[0],)+X.shape[2:]

        if len(X.shape)==4:
            X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2],X.shape[3])

        data_set_x = theano.shared(np.asarray(X,dtype=theano.config.floatX),borrow=True)

        if Y is None:
            pred_model = theano.function(
                            inputs=[index],
                            outputs = self.classifier.probs,
                            givens = {
                                self.input : data_set_x[index*batch_size:(index+1)*batch_size]
                            }
            )

            n_batches = X.shape[0]/batch_size
            elapsed_times = []
            for i in range(n_batches):
                if i == 0:
                    start_time = timeit.default_timer()
                    output = pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                else:
                    start_time = timeit.default_timer()
                    tem_out =  pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                    output = np.concatenate([output,tem_out])

            return output.reshape(actshape),elapsed_times
        else:
            data_set_y = theano.shared(np.asarray(Y,dtype='int32'),borrow=True)

            pred_model = theano.function(
                            inputs=[index],
                            outputs=[self.classifier.probs,self.classifier.cost(self.y),self.classifier.error(self.y)],
                            givens={
                                self.input: data_set_x[index*batch_size:(index+1)*batch_size],
                                self.y: data_set_y[index*batch_size:(1+index)*batch_size]
                            }
            )

            n_batches = X.shape[0]/batch_size
            cost,error = (0.0,0.0)
            elapsed_times = []
            for i in range(n_batches):
                if i==0:
                    start_time = timeit.default_timer()
                    output,cost,error = pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                else:
                    start_time = timeit.default_timer()
                    out,cst,err = pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                    output = np.concatenate([output,out])
                    cost += cst
                    error += err

            return output.reshape(actshape),cost/n_batches,error/n_batches,elapsed_times

    def predict_pgy(self,X,Y=None,mask=None,batch_size=1):
        import timeit
        index = T.iscalar('index')
        self.y = itensor5('y')

        actshape = X.shape + (self.n_out,) if X.ndim == 4 else (X.shape[0],)+X.shape[2:] + (self.n_out,)

        if len(X.shape)==4:
            X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2],X.shape[3])

        data_set_x = theano.shared(np.asarray(X,dtype=theano.config.floatX),borrow=True)

        if Y is None:
            pred_model = theano.function(
                            inputs=[index],
                            outputs = self.classifier.p_given_y,
                            givens = {
                                self.input : data_set_x[index*batch_size:(index+1)*batch_size]
                            }
            )

            n_batches = X.shape[0]/batch_size
            elapsed_times = []
            for i in range(n_batches):
                if i == 0:
                    start_time = timeit.default_timer()
                    output = pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                else:
                    start_time = timeit.default_timer()
                    tem_out =  pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                    output = np.concatenate([output,tem_out])

            return output.reshape(actshape),elapsed_times
        else:
            data_set_y = theano.shared(np.asarray(Y,dtype='int32'),borrow=True)

            pred_model = theano.function(
                            inputs=[index],
                            outputs=[self.classifier.p_given_y,self.classifier.cost(self.y),self.classifier.error(self.y)],
                            givens={
                                self.input: data_set_x[index*batch_size:(index+1)*batch_size],
                                self.y: data_set_y[index*batch_size:(1+index)*batch_size]
                            }
            )

            n_batches = X.shape[0]/batch_size
            cost,error = (0.0,0.0)
            elapsed_times = []
            for i in range(n_batches):
                if i==0:
                    start_time = timeit.default_timer()
                    output,cost,error = pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                else:
                    start_time = timeit.default_timer()
                    out,cst,err = pred_model(i)
                    elapsed_times.append(timeit.default_timer() - start_time)
                    output = np.concatenate([output,out])
                    cost += cst
                    error += err

            return output.reshape(actshape),cost/n_batches,error/n_batches,elapsed_times


    def save_model(self,filename):
        data = []
        for p in self.params:
            data.append(p.get_value())

        try:
            save_data(filename,(self.input_params,data))
            print('model successfully saved!')
        except:
            print('unable to save data!')

    @classmethod
    def load_model(cls,filename):
        try:
            inp,data= load_data(filename)
            n_out,conv_layers = inp
            cl = cls(n_out=n_out,conv_layers=conv_layers)

            for i in range(len(cl.params)):
                cl.params[i].set_value(data[i])

            print('model successfully loaded!')
            return cl
        except:
            print('unable to load model!')
            return None
