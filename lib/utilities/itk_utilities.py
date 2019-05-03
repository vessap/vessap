# -*- coding: utf-8 -*-
"""
Last Modification 09.02.2018

@author: paetzold, alberts,tetteh
"""
import os as os
import SimpleITK as itk
import numpy as np

def make_itk_image(imageArray,protoImage=None):
    ''' Create an itk image given an image numpy ndarray (imageArray) and an
    itk proto-image (protoImage) to provide Origin, Spacing and Direction.'''

    image = itk.GetImageFromArray(imageArray)
    if protoImage != None:
        image.CopyInformation(protoImage)

    return image

def write_itk_image(image,filename):
    ''' Write an itk image to a specified filename.'''

    writer = itk.ImageFileWriter()
    writer.SetFileName(filename)

    if filename.endswith('.nii'):
        Warning('You are converting nii, be careful with type conversions')

    writer.Execute(image)

    return

def get_itk_image(filename):
    ''' Get an itk image given an image filename of extionsion *TIFF, JPEG,
    PNG, BMP, DICOM, GIPL, Bio-Rad, LSM, Nifti, Analyze, SDT/SPR (Stimulate),
    Nrrd or VTK images*.'''

    reader = itk.ImageFileReader()
    reader.SetFileName(filename)

    image = reader.Execute()

    return image

def get_itk_array(filenameOrImage,normalize=False):
    ''' Get an image array given an image filename of extension *TIFF, JPEG,
    PNG, BMP, DICOM, GIPL, Bio-Rad, LSM, Nifti, Analyze, SDT/SPR (Stimulate),
    Nrrd or VTK images*.'''

    if isinstance(filenameOrImage,str):
        image = get_itk_image(filenameOrImage)
    else:
        image = filenameOrImage

    imageArray = itk.GetArrayFromImage(image)
    if normalize:
        imageArray = imageArray - np.min(imageArray)
        imageArray = imageArray*1.0 / np.max(imageArray)

    return imageArray

def write_itk_imageArray(imageArray,filename):
    img = make_itk_image(imageArray)
    write_itk_image(img,filename)

if __name__=="__main__":
    pass
