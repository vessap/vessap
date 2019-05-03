import theano
import theano.tensor as T
import numpy as np

def getTheanoConfusionMatrix(pred_y,data_y,p_values,mask=None):
    """
    :param pred_y: numpy.ndarray of Predicted Values
    :param data_y: numpy.ndarray of Ground truth NOTE should be of the same shape as pred_y
    :param p_values: a list of labels to use as the positive label NOTE in case of multi labels the returned values are averaged
    :param mask: a mask of the same shape as pred_y and or data_y
    :return: List of float values of the form [True Positive,False Positive,False Negative,True Negative]
    """
    pred_y_f = pred_y.flatten()
    data_y_f = data_y.flatten()
    if not mask is None:
        mask_f = mask.flatten()

    TPS = T.constant(0,dtype='float32')
    FPS = T.constant(0,dtype='float32')
    TNS = T.constant(0,dtype='float32')
    FNS = T.constant(0,dtype='float32')

    for p in p_values:
        if mask is None:
            TPS += T.sum(T.eq(pred_y_f,p) * T.eq(data_y_f,p))
            FPS += T.sum(T.eq(pred_y_f,p) * T.neq(data_y_f,p))
            TNS += T.sum(T.neq(pred_y_f,p) * T.neq(data_y_f,p))
            FNS += T.sum(T.neq(pred_y_f,p) * T.eq(data_y_f,p))
        else:
            TPS += T.sum(T.eq(pred_y_f,p) * T.eq(data_y_f,p)*mask_f)
            FPS += T.sum(T.eq(pred_y_f,p) * T.neq(data_y_f,p)*mask_f)
            TNS += T.sum(T.neq(pred_y_f,p) * T.neq(data_y_f,p)*mask_f)
            FNS += T.sum(T.neq(pred_y_f,p) * T.eq(data_y_f,p)*mask_f)


    return [TPS/len(p_values),FPS/len(p_values),FNS/len(p_values),TNS/len(p_values)]

def getTheanoSimilarityScore(pred_y,data_y,p_values,mask=None):
    """
    A a symbolic function for calculating DICE SCORE,Sensitivity and Precision
    :param pred_y: numpy.ndarray of Predicted Values
    :param data_y: numpy.ndarray of Ground truth NOTE should be of the same shape as pred_y
    :param p_values: a list of labels to use as the positive label NOTE in case of multi labels the returned values are averaged
    :param mask: a mask of the same shape as pred_y and or data_y
    :return: List of Float values of the form [DICE SCORE,SENSITIVITY,PRECISION]
    """
    TP,FP,FN,TN = getTheanoConfusionMatrix(pred_y,data_y,p_values,mask)
    prec = TP / (TP + FP+0.000001)
    rec =  TP / (TP + FN+0.000001)
    dice = 2 * (prec*rec) / (prec + rec + 0.000001)
    return dice,rec,prec

