from __future__ import print_function
from dvn.utils import get_itk_array,write_itk_imageArray
import numpy as np
from dvn import FCN, VNET, UNET
from dvn import Network as NET
from dvn import losses as ls
import argparse
import os
import keras as K

# Defined global variables here .................................
BLOCK_SIZE = (64, 64, 64)
OVERLAP_SIZE = (5, 5, 5)
SEL_TOP = 50

def norm_data(data):
    data = data - np.min(data)
    data = data * 1.0 / np.max(data)
    return data

def histinfo(data, cfreq):
    bins = np.arange(np.min(data), np.max(data), 10)
    vals, bins = np.histogram(data, bins, density=True)
    acc = 0
    cutoff = np.max(bins)
    cfreq *= sum(vals)
    for i, v in enumerate(vals):
        acc = acc+v
        if acc >= cfreq:
            cutoff = bins[i]
            break

    return data

def processdata(X, hist_cutoff, n_in):
    if n_in == 1:
        data = histinfo(data=X, cfreq=float(hist_cutoff))
        data = norm_data(data=data)
        return data
    else:
        cfs = hist_cutoff.split(' ')
        if len(cfs) == 1:
            cfs = cfs * len(X)
        assert len(X) == len(cfs), 'Number of channels must match with number of histogram cutoffs'
        data = np.zeros(X.shape, dtype=X.dtype)
        for i in range(len(X)):
            data[i] = norm_data(histinfo(data=X[i], cfreq=float(cfs[i])))

        return data

def load_model(filename=None, n_in=1, n_out=2, modelType=0, use_crosshair=False):
    if filename is None:
        modelType = [FCN, VNET, UNET][modelType]
        model = modelType(nlabels=n_out, nchannels=n_in, cross_hair=use_crosshair)
    else:
        filename = os.path.abspath(filename)
        model = NET.load(filename)

    return model

def train_model(model,data_x,data_y,data_mask,batch_size,n_epochs,lr,weighted_cost,initial_epoch=0):
    k = []
    if weighted_cost:
        for l in np.unique(data_y):
            if l != 0:
                k.append(l)
    sgd = K.optimizers.SGD(lr=lr, decay=0, momentum=0, nesterov=False)
    loss = ls.weighted_categorical_crossentropy_with_fpr(classes=len(k)) if weighted_cost else ls.categorical_crossentropy(axis=1)
    model.compile(optimizer=sgd, loss=loss)
    model.fit(x=data_x,y=data_y,epochs=n_epochs,batch_size=batch_size,initial_epoch=initial_epoch)
    return model, lr

def generate_data(inputFn,labelFn,maskFn,preprocess=False,hist_cutoff=[],n_in=1, cube_size=64):
    iFn = []
    lFn = []
    mFn = []
    with open(os.path.abspath(inputFn)) as f:
        iFn = f.readlines()
    iFn = [x.strip() for x in iFn]

    with open(os.path.abspath(labelFn)) as f:
        lFn = f.readlines()
    lFn = [x.strip() for x in lFn]

    if not maskFn is None:
        with open(os.path.abspath(maskFn)) as f:
            mFn = f.readlines()
        mFn = [x.strip() for x in mFn]

    data_x = []
    data_y = []
    data_mask = []

    if not maskFn is None:
        for ifn,lfn,mfn in zip(iFn,lFn,mFn):
            if n_in == 1:
                X = get_itk_array(ifn)
            else:
                ifns = ifn.split('\t')
                assert len(ifns) >= n_in, 'Number of Input files per line should match the number of channels (tab separated!)'
                X = []
                sh = None
                for ii in ifns[:n_in]:
                    x = get_itk_array(ii)
                    assert (sh is None) or sh == x.shape, 'Input files should have the same shape'
                    X.append(x)

        X = np.array(X).astype('float32')
        Y = get_itk_array(lfn)
        M = get_itk_array(mfn)

        assert X.shape[-3:] == Y.shape
        assert X.shape[-3:] == M.shape

        if preprocess:
            X = processdata(X=X, hist_cutoff=hist_cutoff, n_in=n_in)
            data_x.append(X)
        else:
            data_x.append(X)
        data_y.append(Y)
        data_mask.append(M)
    else:
        for ifn,lfn in zip(iFn,lFn):
            if n_in == 1:
                X = get_itk_array(ifn)
            else:
                ifns = ifn.split('\t')
                assert len(ifns) >= n_in, 'Number of Input files per line should match the number of channels (tab separated!)'
                X = []
                sh = None
                for ii in ifns[:n_in]:
                    x = get_itk_array(ii)
                    assert (sh is None) or sh == x.shape, 'Input files should have the same shape'
                    X.append(x)

        X = np.array(X).astype('float32')
        Y = get_itk_array(lfn)

        assert X.shape[-3:] == Y.shape

        if preprocess:
            X = processdata(X=X, hist_cutoff=hist_cutoff, n_in=n_in)
            data_x.append(X)
        else:
            data_x.append(X)

        data_y.append(Y)

    if not maskFn is None:
        return get_bounding_blocks(np.array(data_x, dtype='float32'),np.array(data_y, dtype='int32'),np.array(data_mask, dtype='int32'),cube_size=cube_size)
    else:
        return get_bounding_blocks(np.array(data_x, dtype='float32'),np.array(data_y, dtype='int32'), None, cube_size=cube_size)

def get_bounding_blocks(feats,grd_truth,mask=None,gmt=False,cube_size=64):
    feat_stack = np.array([])
    grd_truth_stack=np.array([])
    mask_stack = np.array([])
    cnt = 0
    BLOCK_SIZE=[cube_size if s > cube_size else s for s in grd_truth.shape[-3:]]

    for vol_ind in range(grd_truth.shape[0]):
        values = []
        indices = []

    for x in np.arange(0,grd_truth.shape[1]-BLOCK_SIZE[0],BLOCK_SIZE[0]-OVERLAP_SIZE[0]):
        for y in np.arange(0,grd_truth.shape[2]-BLOCK_SIZE[1],BLOCK_SIZE[1]-OVERLAP_SIZE[1]):
            for z in np.arange(0,grd_truth.shape[3]-BLOCK_SIZE[2],BLOCK_SIZE[2]-OVERLAP_SIZE[2]):
                data = grd_truth[vol_ind][x:x+BLOCK_SIZE[0],y:y+BLOCK_SIZE[1],z:z+BLOCK_SIZE[2]]

                if gmt:
                    val = np.sum(data==1) + np.sum(data==2)
                else:
                    val = np.sum(data==1)

                values.append(val)
                indices.append((x,y,z))

    values = np.argsort(np.asarray(values))
    fsh = len(feats.shape) == 5
    for ind in values[values.shape[0]-SEL_TOP:]:
        x,y,z = indices[ind]
        if cnt==0:
            if fsh:
                feat_stack = feats[vol_ind:vol_ind+1,:,x:x+BLOCK_SIZE[0],y:y+BLOCK_SIZE[1],z:z+BLOCK_SIZE[2]]
            else:
                feat_stack = feats[vol_ind:vol_ind+1,x:x+BLOCK_SIZE[0],y:y+BLOCK_SIZE[1],z:z+BLOCK_SIZE[2]]

            grd_truth_stack = grd_truth[vol_ind:vol_ind+1,x:x+BLOCK_SIZE[0],y:y+BLOCK_SIZE[1],z:z+BLOCK_SIZE[2]]

            if not mask is None:
                mask_stack = mask[vol_ind:vol_ind+1,x:x+BLOCK_SIZE[0],y:y+BLOCK_SIZE[1],z:z+BLOCK_SIZE[2]]

        else:
            if fsh:
                fs = feats[vol_ind:vol_ind+1,:,x:x+BLOCK_SIZE[0],y:y+BLOCK_SIZE[1],z:z+BLOCK_SIZE[2]]
            else:
                fs = feats[vol_ind:vol_ind+1,x:x+BLOCK_SIZE[0],y:y+BLOCK_SIZE[1],z:z+BLOCK_SIZE[2]]

            feat_stack = np.concatenate([feat_stack,fs])
            grd_truth_stack = np.concatenate([grd_truth_stack,grd_truth[vol_ind:vol_ind+1,x:x+BLOCK_SIZE[0],y:y+BLOCK_SIZE[1],z:z+BLOCK_SIZE[2]]])

            if not mask is None:
                mask_stack = np.concatenate([mask_stack,mask[vol_ind:vol_ind+1,x:x+BLOCK_SIZE[0],y:y+BLOCK_SIZE[1],z:z+BLOCK_SIZE[2]]])

        cnt += 1

    if mask is None:
        return feat_stack,grd_truth_stack
    else:
        return feat_stack,grd_truth_stack,mask_stack

def parse_args():
    parser = argparse.ArgumentParser(description='Train/Finetune a cross-hair filter based FCN on NIFTI volumes')
    parser.add_argument('--inputFns', dest='inputFns', type=str, default='inputs.txt',
                   help='a text file containing a list of names/path of input data for the traning (one example per line) (default: inputs.txt)')
    parser.add_argument('--labelFns', dest='labelFns', type=str, default='labels.txt',
                   help='a text file containing a list of names/path of data label for the traning (one example per line) (default: labels.txt)')
    parser.add_argument('--maskFns', dest='maskFns', type=str, default=None,
                   help='a text file containing a list of names/path of mask data for the traning (one example per line)')
    parser.add_argument('--preprocess', dest='preprocess', action='store_true',
                   help='Whether to apply preprocessing or not (default: False)')
    parser.add_argument('--hist-cutoff', dest='hist_cutoff', type=str, default="0.99",
                   help='Cutoff to use when applying histogram cutoff (default: 0.99)')
    parser.add_argument('--initModel', dest='model', type=str, default=None,
                   help='a path to a model which should be used as a base for the training (default: None)')
    parser.add_argument('--modelType', dest='modelType', type=int, default=0,
                   help='the model type to train (FCN=0, VNET=1, UNET=2) (default: 0)')
    parser.add_argument('--use_crosshair', dest='usecrosshair', action='store_true',
                   help='Whether to use crosshair filters or not (default: False)')
    parser.add_argument('--n_in', dest='n_in', type=int, default=1,
                   help='number of input channels (default: 1)')
    parser.add_argument('--n_out', dest='n_out', type=int, default=2,
                   help='number of prediction classes (default: 2)')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1,
                   help='batch size for training (default: None)')
    parser.add_argument('--cs', dest='cube_size', type=int, default=64,
                   help='Size of cube to be used during training (default: 64)')
    parser.add_argument('--epochs', dest='epochs', type=int, default=1,
                   help='number of training epochs (default: 1)')
    parser.add_argument('--save-after', dest='save_after', type=int, default=1,
                   help='number of training epochs after which the model should be saved (default: 1)')
    parser.add_argument('--modelFn', dest='modelFn', type=str, default='model',
                   help='filename for saving trained models. Note .dat will be appended autmatically (default: model)')
    parser.add_argument('--modelFolder', dest='model_folder', type=str, default='',
                   help='folder where models will be saved (default: current working directory)')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01,
                   help='learning rate (default: 0.01)')
    parser.add_argument('--decay', dest='decay', type=float, default=0.99,
                   help='learning rate decay per epoch (default: 0.99)')
    parser.add_argument('--weighted-cost', dest='weighted_cost', action='store_true',
                   help='Whether to use weighted cost or not (default: False)')
    args = parser.parse_args()

    return args

def run():
    args = parse_args()
    n_in = args.n_in
    model = load_model(args.model, args.n_in, args.n_out, args.modelType, args.usecrosshair)
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.learning_rate
    decay = args.decay
    weighted_cost = args.weighted_cost
    model_folder = args.model_folder
    modelFn = args.modelFn
    save_after = args.save_after

    if args.maskFns is None:
        data_x, data_y = generate_data(args.inputFns, args.labelFns, args.maskFns, preprocess=args.preprocess, hist_cutoff=args.hist_cutoff, n_in=args.n_in,cube_size=args.cube_size)
        sh = data_y.shape
        data_y = data_y.reshape((sh[0], 1) + sh[1:])
        if n_in == 1:
            data_x = data_x.reshape((sh[0], 1) + sh[1:])
        data_mask = None
    else:
        data_x, data_y, data_mask = generate_data(args.inputFns, args.labelFns, args.maskFns, preprocess=args.preprocess, hist_cutoff=args.hist_cutoff, n_in=args.n_in, cube_size=args.cube_size)
        sh = data_y.shape
        data_y = data_y.reshape((sh[0], 1) + sh[1:])
        data_mask = data_mask.reshape((sh[0], 1) + sh[1:])
        if n_in == 1:
            data_x = data_x.reshape((sh[0], 1) + sh[1:])

    n_iters = int(n_epochs / save_after)
    iters = [save_after for i in range(n_iters)]

    if n_epochs % save_after > 0:
        iters.append(n_epochs % save_after)

    sh = data_x.shape
    print('..............................')
    print('Training Parameters')
    print('..............................')
    print('learning-rate:',lr)
    print('decay:', decay)
    print('Number of epochs:', n_epochs)
    print('Batch size:', batch_size)
    print('Weighted cost:',weighted_cost)
    print('Base Model:',args.model)
    print('Model save folder:', model_folder)
    print('Model save filename:',modelFn)
    print('save model after every', save_after, 'epoch(s)')
    print('Apply preprocessing:', args.preprocess)
    print('Histogram cutoff:', args.hist_cutoff)
    print('Number of input channels:', args.n_in)
    print('Number of classes:', args.n_out)
    print('Number of Examples Extracted:', sh[0])
    print('Training cube size:', sh[-3:])
    print('...................................................\n \n')

    for i, this_epochs in enumerate(iters):
        model, lr = train_model(model=model,data_x=data_x,data_y=data_y,data_mask=data_mask,batch_size=batch_size,n_epochs=this_epochs,lr=lr,weighted_cost=weighted_cost)
        this_model_fn = os.path.abspath(os.path.join(model_folder,'model'+str(i+1)+'.dat'))
        print('saving model......')
        model.save(this_model_fn)
        lr = lr * decay
        print('.....................................................................')

if __name__ == "__main__":
    run()
