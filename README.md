unet_nuclei
===========



Overview
--------

This package provides stand-alone version of unet segmentation of fluorescent nuclei based on the code from the Cellprofiler plugin 
https://github.com/CellProfiler/CellProfiler-plugins/blob/master/classifypixelsunet.py

You can use the pre-trained Unet to segment fluorescece microsopy images of cell nuclei. The output will be a three-channel classification result that provides the probabilities for a pixel to
be part of a background, nucleus and boundary pixel. The seperate boundary class is useful for splitting clumped nuclei.


I mainly created the package for my personal use, so I can use the very useful nuclei segmentation in my own python scripts without CellProfiler as a dependency. I'm only releasing this in case other people find it useful. 

Installation / Usage
--------------------

To install use pip:

    $ pip install git+https://gitlab.erc.monash.edu.au/mmi/unet-nuclei.git


Or clone the repo:

    $ git clone https://gitlab.erc.monash.edu.au/mmi/unet-nuclei.git
    $ python setup.py install
    

Keras Backend and GPU support
-----------------------------

The unet is implemented using Keras and works with both tensorflow as an `cntk` as backend. You can set the behaviour by setting the environment variable `KERAS_BACKEND` to either `tensorflow` or `cntk`. Other options to set the backend are described here: https://keras.io/backend/.

Depending on whether you use `conda` or `pip` you should be able to install tensorflow and cntk with either of those.

If you have a suitable GPU you may want to install a GPU-accelerated versions of the desired backend. If you use conda, you should be able to do `conda install tensorflow-gpu`.

For installing `cntk` with GPU support it is probably the easiest to download a wheel file from https://docs.microsoft.com/en-us/cognitive-toolkit/setup-windows-python?tabs=cntkpy26#2-install-from-wheel-files and install it using `pip`.


Usage and examples
------------------


In the simplest case:

```
from unet_nuclei import *
import skimage.io

# Red
img


```

See the example Jupyter notebook under `./notebook`.


Credit
------

Credits are due to the Cellprofiler team for training the unet and providing an implementation with trained weights.

Contains code from various authors, see LICENSE.md for license and authors.

Package maintainer: Volker Hilsenstein


Contributing
------------

Please create a Github issue for any bug reports and improvements.
