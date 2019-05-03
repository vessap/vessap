import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse

def binary_entropy(grd_truth,predicted):
    L = -T.sum(grd_truth*T.log(predicted) + (1-grd_truth)*T.log(1-predicted),axis=-1)
    return T.mean(L)


def categorical_entropy(predicted,mask=None):
    if mask is None:
        L = -T.mean(T.log(predicted))
        return L
    else:
        L = -(mask.flatten()*T.log(predicted.flatten()))
        return T.sum(L)*1.0/(T.sum(mask)+1)


def categorical_weighted_entropy(grd_truth,predicted,classes,bin_pred=None,mask=None):
        
    cost = T.constant(0.0)
    if mask is None:
	log_out = T.log(predicted.flatten())
        for i in classes:
            L = -(log_out*T.eq(grd_truth.flatten(),i))
	    den = T.sum(T.eq(grd_truth,i))
            cost += T.sum(L)/(den+1)
	     
	    if not bin_pred is None:
		f_pos = T.neq(grd_truth.flatten(),i)*T.eq(bin_pred.flatten(),i) 
		w_pred = -(log_out)*f_pos
                w_weight = 0.5# + T.sum(T.abs_(predicted.flatten() - 0.5)*f_pos)/(T.sum(f_pos)+1)
		cost += (T.sum(w_pred)/(den+1))*w_weight
	    
        return cost*1.0/len(classes)

    else:
        log_out = T.log(predicted.flatten())*mask.flatten()
        denom=T.sum(mask)
        for i in classes:
            cl_vals = T.eq(grd_truth.flatten(),i) 
            L = -(log_out*cl_vals)
	    den = T.sum(mask.flatten()*cl_vals)
            cost += T.sum(L)*1.0/(den+1)
	    
	    if not bin_pred is None:
		f_pos = T.neq(grd_truth.flatten(),i)*T.eq(bin_pred.flatten(),i)*mask.flatten()
                w_pred = -(log_out)*f_pos
                w_weight = 0.5# + T.sum(T.abs_(predicted.flatten() - 0.5)*f_pos)/(T.sum(f_pos)+1)
                cost += (T.sum(w_pred)/(den+1))*w_weight
	    
        return cost*1.0/len(classes) 

def new_categorical_weighted_entropy(grd_truth,predicted,classes,bin_pred=None,mask=None):
    cost = T.constant(0.0)
    tot_weight = T.constant(0.0)
    if mask is None:
        log_out = T.log(predicted.flatten())
        total_labels = T.sum(T.eq(grd_truth,0)) + T.sum(T.neq(grd_truth,0))
        for i in classes:
            L = -(log_out*T.eq(grd_truth.flatten(),i))
            den = T.sum(T.eq(grd_truth,i))
            cost += T.sum(L) * (1.0 - (den/total_labels))
	    tot_weight += (1.0 - (den/total_labels)) 
	    '''
            if not bin_pred is None:
                f_pos = T.neq(grd_truth.flatten(),i)*T.eq(bin_pred.flatten(),i)
                w_pred = -(log_out)*f_pos
                w_weight = 0.5 + T.sum(T.abs_(predicted.flatten() - 0.5)*f_pos)/(T.sum(f_pos)+1)
                cost += (T.sum(w_pred)/(den+1))*w_weight
	   '''
        return cost / tot_weight

    else:
        log_out = T.log(predicted.flatten())*mask.flatten()
        total_labels = T.sum(mask)
        for i in classes:
            cl_vals = T.eq(grd_truth.flatten(),i)
            L = (log_out*cl_vals)
            den = T.sum(mask.flatten()*cl_vals)
            cost += -T.sum(L)*(1.0-(den/total_labels))
	    tot_weight += (1.0-(den/total_labels))
	    '''
            if not bin_pred is None:
                f_pos = T.neq(grd_truth.flatten(),i)*T.eq(bin_pred.flatten(),i)*mask.flatten()
                w_pred = -(log_out)*f_pos
                w_weight = 0.5  + T.sum(T.abs_(predicted.flatten() - 0.5)*f_pos)/(T.sum(f_pos)+1)
                cost += T.sum(w_pred) * ((1-(den/total_labels))*w_weight)
	    '''

        return cost / (tot_weight*len(classes))



def mean_squared_error(grd_truth,predicted,mask=None):
    if mask is None:
        return T.mean((grd_truth.flatten()-predicted.flatten())**2)
    else:
        L = T.sum(((grd_truth.flatten()-predicted.flatten())*mask.flatten())**2)
        return T.sum(L)*1.0/(T.sum(mask)+0.005)

