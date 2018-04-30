### About the MNIST Dataset

The dataset is well known I guess due to great Yann LeCun and all unnecessary information can be found [here](http://yann.lecun.com/exdb/mnist/).  Still if you are wondering about the dataset, here it is :

![Image result for mnist dataset](https://www.researchgate.net/profile/Steven_Young11/publication/306056875/figure/fig1/AS:393921575309346@1470929630835/Example-images-from-the-MNIST-dataset.ppm)



### Goal of this implementation

Our aim should be to implement a simple generative network based on GANs to train on MNIST dataset and then generate the images. 



### Implementation

###### Lets start with the main function

we should create structure of how our high level pipeline should look like:

```python
def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = # TODO

    # Build model.
    model = Gan() # TODO

    # Start training
    train() # TODO
    
    # Generate samples
    generate() # TODO


if __name__ == "__main__":
    tf.app.run()
```

So this will be our `main.py` file. 

###### Now lets move on to getting data

We are lucky that `tensorflow` provides the data in tensor format and we just have to use the following lines of code. 



```python 
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Functions for downloading and reading MNIST data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
```

we just literally need to paste it one of the file and lets call it `input_data.py`. Getting data is done !

###### Update the main function

```python
import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan() # TODO

    # Start training
    train() # TODO
    
    # Generate samples
    generate() # TODO


if __name__ == "__main__":
    tf.app.run()
```

###### Lets create the GAN class

lets create a `constructor` (what we do usually !) and create a `generator` function and a `discriminator` function. We might also need two more functions for `discriminator loss` and `generator loss` . Lets create one function for `generating samples` as well (Not sure if we need it or not!)

```python
class Gan(object):
    """Adversary based generator network.
    """

    def __init__(self):
        """Initializes a GAN"""
		pass

    def _discriminator(self, x, reuse=False):
        """Discriminator block of the network"""
        
		y = None
        return y

    def _discriminator_loss(self, y, y_hat):
        """Loss for the discriminator."""

        l = None
        return l

    def _generator(self, z, reuse=False):
        """From a sampled z, generate an image."""
			x_hat = None
            return x_hat

    def _generator_loss(self, y_hat):
        """Loss for the discriminator."""
        
        l = None
        return l

    def generate_samples(self, z_np):
        """Generates random samples from the provided z_np."""
        out = None
        return out

```

save this in a folder `models/gan.py` 