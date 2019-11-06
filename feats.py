from __future__ import print_function
import argparse
import os
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy import ndimage as ndi
from lib.dvn.utils import get_itk_array, make_itk_image, write_itk_image, get_itk_image

def parse_args():
    parser = argparse.ArgumentParser(description='Extract Centerlines, Bifurcations and Radius from binary vessel segmentation')
    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',
                   help='input filename(s) should follow the sequence for multiple channels')
    parser.add_argument('--output', dest='output', type=str, default='',
                   help='output folder for storing predictions (default: current working directory)')
    parser.add_argument('--no-c', dest='save_centerlines', action='store_false',
                   help='Do not save centerline extraction')
    parser.add_argument('--o_cens', dest='suffix_cens', type=str, default='_cens',
                   help='filename suffix for renaming CENTERLINE output files (default: _cens)')
    parser.add_argument('--no-b', dest='save_bifurcations', action='store_false',
                   help='Do not save bifurcation detection')
    parser.add_argument('--o_bifs', dest='suffix_bifs', type=str, default='_bifs',
                   help='filename suffix for renaming BIFURCATION output files (default: _bifs)')
    parser.add_argument('--no-r', dest='save_radius', action='store_false',
                   help='Do not save radius estimates')
    parser.add_argument('--o_rads', dest='suffix_rads', type=str, default='_rads',
                   help='filename suffix for renaming RADIUS output files (default: _rads)')
    parser.add_argument('--f', dest='format', type=str, default='.nii.gz',
                   help='NIFTI file format for saving outputs (default: .nii.gz)')
    args = parser.parse_args()

    return args

def extract_centerlines(segmentation):
    skeleton = skeletonize_3d(segmentation)
    skeleton.astype(dtype='uint8', copy=False)
    return skeleton

def extract_bifurcations(centerlines):
    a = centerlines
    a.astype(dtype='uint8', copy=False)
    sh = np.shape(a)
    bifurcations = np.zeros(sh,dtype='uint8')
    endpoints = np.zeros(sh,dtype='uint8')

    for x in range(1,sh[0]-1):
        for y in range(1,sh[1]-1):
            for z in range(1,sh[2]-1):
                if a[x,y,z]== 1:
                    local = np.sum([a[ x-1,  y-1,  z-1],
                    a[ x-1,  y-1,  z],
                    a[ x-1,  y-1,  z+1],
                    a[ x-1,  y,  z-1],
                    a[ x-1,  y,  z],
                    a[ x-1,  y,  z+1],
                    a[ x-1,  y+1,  z-1],
                    a[ x-1,  y+1,  z],
                    a[ x-1,  y+1,  z+1],
                    a[ x,  y-1,  z-1],
                    a[ x,  y-1,  z],
                    a[ x,  y-1,  z+1],
                    a[ x,  y,  z-1],
                    a[ x,  y,  z],
                    a[ x,  y,  z+1],
                    a[ x,  y+1,  z-1],
                    a[ x,  y+1,  z],
                    a[ x,  y+1,  z+1],
                    a[ x+1,  y-1,  z-1],
                    a[ x+1,  y-1,  z],
                    a[ x+1,  y-1,  z+1],
                    a[ x+1,  y,  z-1],
                    a[ x+1,  y,  z],
                    a[ x+1,  y,  z+1],
                    a[ x+1,  y+1,  z-1],
                    a[ x+1,  y+1,  z],
                    a[ x+1,  y+1,  z+1]])

                    if local > 3:
                        bifurcations[x,y,z] = 1

    bifurcations.astype(dtype='uint8', copy=False)
    endpoints.astype(dtype='uint8', copy=False)
    return bifurcations, endpoints

def extract_radius(segmentation, centerlines):
    image = segmentation
    skeleton = centerlines
    transf = ndi.distance_transform_edt(image,return_indices=False)
    radius_matrix = transf*skeleton
    return radius_matrix

def preprocess_data(data):
    data = data.astype(np.int)
    data = ndi.binary_closing(data, iterations=1).astype(np.int)
    data = np.asarray(ndi.binary_fill_holes(data), dtype='uint8')
    return data

def save_data(data, img, filename):
    out_img = make_itk_image(data, img)
    write_itk_image(out_img, filename)

def run():
    args = parse_args()
    filenames = args.filenames
    outputFn = args.output
    save_cen = args.save_centerlines
    save_bif = args.save_bifurcations
    save_rad = args.save_radius
    cen_suffix = args.suffix_cens
    bif_suffix = args.suffix_bifs
    rad_suffix = args.suffix_rads
    fmt = args.format

    print ('----------------------------------------')
    print (' Feature Extraction Parameters ')
    print ('----------------------------------------')
    print ('Input files:', filenames)
    print ('Output folder:', outputFn)
    print ('Output format:', fmt)
    print ('Centerline file suffix:', cen_suffix)
    print ('Bifurcation file suffix:', bif_suffix)
    print ('Radius file suffix:', rad_suffix)
    print ('Save centerline extractions:', save_cen)
    print ('Save bifurcation detections:', save_bif)
    print ('Save radius estimates:', save_rad)
    print ('----------------------------------------')

    for fn in filenames:
        print('predicting features for :', fn)
        cen = None
        bif = None
        rad = None
        data = preprocess_data(get_itk_array(fn))
        img = get_itk_image(fn)
        prefix = os.path.basename(fn).split('.')[0]

        if save_rad:
            cen = extract_centerlines(segmentation=data)
            rad = extract_radius(segmentation=data, centerlines=cen)
            ofn = os.path.join(outputFn, prefix + rad_suffix + fmt)
            save_data(data=rad, img=img, filename=ofn)

        if save_bif:
            if cen is None:
                cen = extract_centerlines(segmentation=data)

            bif,_ = extract_bifurcations(centerlines=cen)
            ofn = os.path.join(outputFn, prefix + bif_suffix + fmt)
            save_data(data=np.asarray(bif, dtype='uint8'), img=img, filename=ofn)

        if save_cen:
            if cen is None:
                cen = extract_centerlines(segmentation=data)

            ofn = os.path.join(outputFn, prefix + cen_suffix + fmt)
            save_data(data=np.asarray(cen, dtype='uint8'), img=img, filename=ofn)

    print ('finished!')

if __name__ == '__main__':
    run()
