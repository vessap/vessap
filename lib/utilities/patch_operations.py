# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:59:13 2018

@author: giles-cagdas
"""
import numpy as np

def get_patch_data(volume4d, divs = (2,2,2,1), offset=(5,5,5,0)):
    patches = []
    shape = volume4d.shape
    widths = [ int(s/d) for s,d in zip(shape, divs)]
    patch_shape = [w+o*2 for w, o in zip(widths,offset)]

    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                for t in np.arange(0, shape[3], widths[3]):
                    patch = np.zeros(patch_shape, dtype=volume4d.dtype)

                    x_s = x - offset[0] if x - offset[0] >= 0 else 0
                    x_e = x + widths[0] + offset[0] if x + widths[0] + offset[0] <= shape[0] else shape[0]

                    y_s = y - offset[1] if y - offset[1] >= 0 else 0
                    y_e = y + widths[1] + offset[1] if y + widths[1] + offset[1] <= shape[1] else shape[1]

                    z_s = z - offset[2] if z - offset[2] >= 0 else 0
                    z_e = z + widths[2] + offset[2] if z + widths[2] + offset[2] <= shape[2] else shape[2]

                    t_s = t - offset[3] if t - offset[3] >= 0 else 0
                    t_e = t + widths[3] +  offset[3] if t + widths[3] + offset[3] <= shape[3] else shape[3]

                    vp = volume4d[x_s:x_e,y_s:y_e,z_s:z_e,t_s:t_e]

                    px_s = offset[0] - (x - x_s)
                    px_e = px_s + (x_e - x_s)

                    py_s = offset[1] - (y - y_s)
                    py_e = py_s + (y_e - y_s)

                    pz_s = offset[2] - (z - z_s)
                    pz_e = pz_s + (z_e - z_s)

                    pt_s = offset[3] - (t - t_s)
                    pt_e = pt_s + (t_e - t_s)

                    patch[px_s:px_e,py_s:py_e,pz_s:pz_e,pt_s:pt_e] = vp

                    patches.append(patch)

    return np.array(patches, dtype=volume4d.dtype)

def get_volume_from_patches(patches5d, divs = (2,2,2,1), offset=(5,5,5,0)):
    new_shape = [(ps-of*2)*d for ps,of,d in zip(patches5d.shape[-4:],offset,divs)]
    volume4d = np.zeros(new_shape, dtype=patches5d.dtype)

    shape = volume4d.shape
    widths = [ int(s/d) for s,d in zip(shape,divs)]

    index = 0
    for x in np.arange(0, shape[0], widths[0]):
        for y in np.arange(0, shape[1], widths[1]):
            for z in np.arange(0, shape[2], widths[2]):
                for t in np.arange(0, shape[3], widths[3]):
                    patch = patches5d[index]
                    index = index + 1
                    volume4d[x:x+widths[0],y:y+widths[1],z:z+widths[2],t:t+widths[3]] = patch[offset[0]:offset[0]+widths[0],offset[1]:offset[1]+widths[1],offset[2]:offset[2]+widths[2],offset[3]:offset[3]+widths[3]]

    return volume4d
