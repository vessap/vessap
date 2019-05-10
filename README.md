## README 

For the Nature Methods submission "Automated analysis of whole brain vasculature using machine learning"

Authors:  Johannes C. Paetzold and Giles Tetteh

This code can be executed in a public code ocean capsule at xxxxlink.

## Table of contents

* [General info](#general-info)
* [Segmenting data](#test)
* [Training a model](#train)
* [Feature extraction](#feats)
* [Dependencies](#depend)

## General info

For each of the described tasks, segmentation, training and feature extraction we have  set up a working example (demo) in the code ocean compute capsule. 

1. 	A utility library for the Deep Learning part of the work in THEANO including 
	
	a. The complete theano library including all code, operators, networks and utilities
	
	b. Detailed descriptions how to setup this framework (Readme.md).
	
	c. An instruction file (demo) how to segment your own data and how to train your own model (demo_instructions.md).

2. 	A data folder containing:
	
	a. The synthetic dataset used to pretrain our network
	
	b. The training data and corresponding ground truth annotations
	
	c. A test set 

3. 	A model folder containing:
	
	a. Models trained on sythetic data for 1 and 2 channel network input
	
	b. The fully refined 2 input channel network


## Segmenting data

The prediction on our models generates a binary segmentation (threshold=0.5) and a probabilistic prediction.

Here we provide three models:

* Synthetic model for one chanel input images  ('synth_model.dat'). 
* Synthetic model for two chanel input images  ('synth_model_2ch_input.dat'). 
* Real model from the Vessap paper for two input chanels  ('trained_model.dat').

The following arguments can be passed through the terminal:

    -- help
    
    parser = argparse.ArgumentParser(description='Apply Cross-hair filters network to predict a NIFTI volume')
    
    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+', help='input filename(s) should follow the sequence for multiple channels')
    
    parser.add_argument('--maskFilename', dest='maskFn', type=str, default=None, help='a mask file to be applied to the predictions')
    
    parser.add_argument('--output', dest='output', type=str, default='',help='output folder for storing predictions (default: current working directory)')
    
    parser.add_argument('--o_probs', dest='suffix_probs', type=str, default='_probs',help='filename suffix for renaming probability output files (default: _probs)')
    
    parser.add_argument('--o_bins', dest='suffix_bins', type=str, default='_bins',help='filename suffix for renaming binary output files (default: _bins)')
    
    parser.add_argument('--t', dest='threshold', type=float, default=0.5,help='threshold for converting probabilities to binary (default: 0.5)')
    
    parser.add_argument('--f', dest='format', type=str, default='.nii.gz',help='NIFTI file format for saving outputs (default: .nii.gz)')
    
    parser.add_argument('--model', dest='model', type=str, default='model.dat',help='a saved model file (default: model.dat)')
    
    parser.add_argument('--preprocess', dest='preprocess', action='store_true',help='Whether to apply preprocessing or not')
    
    parser.add_argument('--bs', dest='batch_size', type=int, default=1,help='Batch size to apply during prediction (default: 1)')
    
    parser.add_argument('--cs', dest='cube_size', type=int, default=64,help='Size of cube to be applied during prediction (default: 64)')
    
    parser.add_argument('--hist-cutoff', dest='hist_cutoff', type=str, default="0.99", help='Cutoff to use when applying histogram cutoff (default: 0.99)')
    

## Training a model

Training recquires a labeled dataset, in this git we provide an exemplary dataset with labels. You can either refine a model, for example our 'synth_model' or retrain a model from a random initialization. Please then specify your training set and your labels and pass them as arguments. Further arguments can be passed through the terminal:

    -- help
    
    parser = argparse.ArgumentParser(description='Train/Finetune a cross-hair filter based FCN on NIFTI volumes')
    
    parser.add_argument('--inputFns', dest='inputFns', type=str, default='inputs.txt', help='a text file containing a list of names/path of input data for the traning (one example per line) (default: inputs.txt)') 
    
    parser.add_argument('--labelFns', dest='labelFns', type=str, default='labels.txt', help='a text file containing a list of names/path of data label for the traning (one example per line) (default: labels.txt)')
    
    parser.add_argument('--maskFns', dest='maskFns', type=str, default=None, help='a text file containing a list of names/path of mask data for the traning (one example per line)')
    
    parser.add_argument('--preprocess', dest='preprocess', action='store_true', help='Whether to apply preprocessing or not (default: False)')
    
    parser.add_argument('--hist-cutoff', dest='hist_cutoff', type=str, default="0.99", help='Cutoff to use when applying histogram cutoff (default: 0.99)')
    
    parser.add_argument('--initModel', dest='model', type=str, default=None, help='a path to a model which should be used as a base for the training (default: None)')
    
    parser.add_argument('--n_in', dest='n_in', type=int, default=1, help='number of input channels (default: 1)')
    
    parser.add_argument('--n_out', dest='n_out', type=int, default=2, help='number of prediction classes (default: 2)')
    
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='batch size for training (default: None)')
    
    parser.add_argument('--cs', dest='cube_size', type=int, default=64, help='Size of cube to be used during training (default: 64)')
    
    parser.add_argument('--epochs', dest='epochs', type=int, default=1, help='number of training epochs (default: 1)')
    
    parser.add_argument('--save-after', dest='save_after', type=int, default=1, help='number of training epochs after which the model should be saved (default: 1)')
    
    parser.add_argument('--modelFn', dest='modelFn', type=str, default='model', help='filename for saving trained models. Note .dat will be appended autmatically (default: model)')
    
    parser.add_argument('--modelFolder', dest='model_folder', type=str, default='', help='folder where models will be saved (default: current working directory)')
    
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.01, help='learning rate (default: 0.01)')
    
    parser.add_argument('--decay', dest='decay', type=float, default=0.99,help='learning rate decay per epoch (default: 0.99)')
    
    parser.add_argument('--weighted-cost', dest='weighted_cost', action='store_true', help='Whether to use weighted cost or not (default: False)')

## Feature extraction

The feature extraction extracts the skeleton length, number of bifurcation points, maximum radius and average radius.

#### Use your own data

To extract features from your own images, please have segmented data in an itk-comaptible format first. If you do not have a binary segmentation please run the the [Segmenting data](#test) routine on your images first. The following arguments can be passed through the terminal:

    -- help
    parser = argparse.ArgumentParser(description='Extract Centerlines, Bifurcations and Radius from binary vessel segmentation')
        
    parser.add_argument('filenames', metavar='filenames', type=str, nargs='+',help='input filename(s) should follow the sequence for multiple channels')
    
    parser.add_argument('--output', dest='output', type=str, default='',help='output folder for storing predictions (default: current working directory)')
    
    parser.add_argument('--no-c', dest='save_centerlines', action='store_false',help='Do not save centerline extraction')
    
    parser.add_argument('--o_cens', dest='suffix_cens', type=str, default='_cens',help='filename suffix for renaming CENTERLINE output files (default: _cens)')
                   
    parser.add_argument('--no-b', dest='save_bifurcations', action='store_false', help='Do not save bifurcation detection')
                   
    parser.add_argument('--o_bifs', dest='suffix_bifs', type=str, default='_bifs',help='filename suffix for renaming BIFURCATION output files (default: _bifs)')
                   
    parser.add_argument('--no-r', dest='save_radius', action='store_false',help='Do not save radius estimates')
                   
    parser.add_argument('--o_rads', dest='suffix_rads', type=str, default='_rads',help='filename suffix for renaming RADIUS output files (default: _rads)')
                   
    parser.add_argument('--f', dest='format', type=str, default='.nii.gz', help='NIFTI file format for saving outputs (default: .nii.gz)')
    
#### Regional Features

To extract features in your own images for a particular region of interest, for example regions from the Allen brain atlas, you can upload images of those regions, segment them using the [Segmenting data](#test) routine and then extract the featues using the feature extraction routine. If you have a dataset which is registered to the Allen brain atlas, the utility scripts in the matlab folder allow the calculation of whole brain statistics. 



## Dependencies

* numpy==1.12.0 
* scipy==0.18.1 
* skimage==0.15.x
* Theano==0.9.0b1
