from lib.utilities import get_itk_array, get_itk_image, make_itk_image, write_itk_image, load_data, get_patch_data, get_volume_from_patches
import numpy as np
from lib.networks import FCNN as NET
import argparse
import os

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
        return norm_data(data=data)
    else:
        cfs = hist_cutoff.split(' ')
        if len(cfs) == 1:
            cfs = cfs * len(X)

        assert len(X) == len(cfs), 'Number of channels must match with number of histogram cutoffs'
        data = np.zeros(X.shape, dtype=X.dtype)
        for i in range(len(cfs)):
            data[i] = norm_data(histinfo(data=X[i], cfreq=float(cfs[i])))
        return data

def load_model(filename):
    filename = os.path.abspath(filename)
    model = NET.load_model(filename)
    return model

def generate_data(filenames, maskFn=None, preprocess=False, hist_cutoff="0.99"):
    data = []
    for fn in filenames:
        data.append(get_itk_array(os.path.abspath(fn)))


    data = np.array(data)
    print 'Volume size: ', data.shape[-3:]
    if not maskFn is None:
        mask = get_itk_array(os.path.abspath(maskFn))
        assert mask.shape == data.shape[-3:], 'Image size should match mask size'
    else:
        mask = None

    if preprocess:
        data = processdata(X=data, hist_cutoff=hist_cutoff, n_in=len(filenames))
    return np.asarray(data, dtype='float32'), mask

def parse_args():
    parser = argparse.ArgumentParser(description='Apply Cross-hair filters network to predict a NIFTI volume')
    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',
                   help='input filename(s) should follow the sequence for multiple channels')
    parser.add_argument('--maskFilename', dest='maskFn', type=str, default=None,
                   help='a mask file to be applied to the predictions')
    parser.add_argument('--output', dest='output', type=str, default='',
                   help='output folder for storing predictions (default: current working directory)')
    parser.add_argument('--o_probs', dest='suffix_probs', type=str, default='_probs',
                   help='filename suffix for renaming probability output files (default: _probs)')
    parser.add_argument('--o_bins', dest='suffix_bins', type=str, default='_bins',
                   help='filename suffix for renaming binary output files (default: _bins)')
    parser.add_argument('--t', dest='threshold', type=float, default=0.5,
                   help='threshold for converting probabilities to binary (default: 0.5)')
    parser.add_argument('--f', dest='format', type=str, default='.nii.gz',
                   help='NIFTI file format for saving outputs (default: .nii.gz)')
    parser.add_argument('--model', dest='model', type=str, default='model.dat',
                   help='a saved model file (default: model.dat)')
    parser.add_argument('--preprocess', dest='preprocess', type=int, default=0,
                   help='Whether to apply preprocessing or not (default: 0 => False)')
    parser.add_argument('--bs', dest='batch_size', type=int, default=1,
                   help='Batch size to apply during prediction (default: 1)')
    parser.add_argument('--cs', dest='cube_size', type=int, default=64,
                   help='Size of cube to be applied during prediction (default: 64)')
    parser.add_argument('--hist-cutoff', dest='hist_cutoff', type=str, default="0.99",
                   help='Cutoff to use when applying histogram cutoff (default: 0.99)')
    args = parser.parse_args()

    return args

def predict_volume(model, data, mask, cube_size=64, batch_size=1, cutoff=0.5):
    sh = data.shape
    if len(sh) == 3:
        data = data.reshape((1,)+sh)


    rems = [s%cube_size for s in data.shape[-3:]]
    padding = [cube_size - r if r > 0 else 0 for r in rems]
    if np.sum(padding) > 0:
        nsh = [data.shape[0],] + [s + p for s,p in zip(data.shape[-3:], padding)]
        dummy = np.zeros(nsh, dtype=data.dtype)
        dummy[:,padding[0]:,padding[1]:,padding[2]:] = data
        data = dummy

    sh = data.shape
    divs = [1,] + [(s/cube_size) for s in sh[-3:]]
    patches = get_patch_data(data, divs=divs, offset=(0,5,5,5))
    p,_ = model.predict_pgy(patches, batch_size=batch_size)
    p = np.transpose(p, axes=(0,4,1,2,3))
    volume = get_volume_from_patches(p, divs=divs, offset=(0,5,5,5))

    if np.sum(padding) > 0:
        volume = volume[:,padding[0]:,padding[1]:,padding[2]:]

    if len(volume) == 2:
        probs = volume[1]
        bin_pred = np.asarray(probs >= cutoff, dtype='uint8')
        if not mask is None:
            probs = probs * mask
            bin_pred = bin_pred * mask
    else:
        probs = volume
        bin_pred = np.asarray(np.argmax(volume, axis=0), dtype='uint8')
        if not mask is None:
            bin_pred = bin_pred * mask

    return probs, bin_pred


def run():
    args = parse_args()
    print '----------------------------------------'
    print ' Testing Parameters '
    print '----------------------------------------'
    print 'Model filename:', args.model
    print 'Input files:', args.filenames
    print 'Mask file:', args.maskFn
    print 'Preprocess:', args.preprocess
    print 'Histogram cutoff:', args.hist_cutoff
    print 'Cube size:', args.cube_size
    print 'Batch size:', args.batch_size
    print 'Binary threshold:', args.threshold
    print 'Output folder:', args.output
    print 'Output format:', args.format
    print 'Binary file suffix:', args.suffix_bins
    print 'Probabilities file suffix:', args.suffix_probs
    print '----------------------------------------'

    model = load_model(args.model)
    data, mask = generate_data(args.filenames, args.maskFn, args.preprocess, args.hist_cutoff)
    probs, bins = predict_volume(model=model, data=data, mask=mask, cube_size=args.cube_size, batch_size=args.batch_size, cutoff=args.threshold)

    filename = args.filenames[0]
    img = get_itk_image(filename)
    prefix = os.path.basename(filename).split('.')[0]
    bins = make_itk_image(bins)
    write_itk_image(bins, os.path.join(args.output, prefix + args.suffix_bins + args.format))
    if probs.ndim == 3:
        probs = make_itk_image(probs, img)
        write_itk_image(probs, os.path.join(args.output, prefix + args.suffix_probs + args.format))
    else:
        for i in range(len(probs)):
            d = probs[i]
            d = make_itk_image(d, img)
            write_itk_image(d, os.path.join(args.output, prefix + args.suffix_probs + '_'+ str(i) + args.format))

    print 'finished!'

if __name__ == "__main__":
    run()
