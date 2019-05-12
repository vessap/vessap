

This repository consist of 2 main python modules and a utilities module

### operators ###
The operators module containes classes for constructing the layers in the deep learning networks
e.g. 
1. Convolution operators (Conv3dLayer,Conv2dLayer, Deconv2dLayer)
2. Downsampling Operators
3. RBM, Autoencoders etc.

You might not need this module if you intend to use any of the already existing networks below. However in case you want to 
construct your own network then you can use these operators to build.

### networks ###
This module contains our proposed FCN with Crosshair filters as convolutions

Each of the networks has a built-in saving and loading methods for saving the trained models and loading them later.

### utilities ###
This module contains utility functions for loading, writing and making ITK images e.g.
(Thanks to Esther for contributing some of these codes)
1. get_itk_image
2. get_itk_array
3. write_itk_image
4. make_itk_image
5. etc.

### How do I get set up? ###
Dependencies : 
numpy==1.12.0
scipy==0.18.1
Theano==0.9.0b1
SimpleITK==0.10.0
Pillow==4.0.0

It is better to setup the library in a virtual environment as follows
In your home directory

1. virtualenv theano-env
2. source theano-env/bin/activate
3. cd path/to/downloaded/repository
4. python setup.py install   (This should install the modules into your virtualenv)

## the installation takes less then 5 minutes if all dependencies are installed

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###
For more explanation on how to use any particular part of the code you can contact
* Giles Tetteh (giles.tetteh@tum.de)
* Johannes C. Paetzold (johannes.paetzold@tum.de)
