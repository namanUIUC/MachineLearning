"""Implements feature extraction and other data processing helpers.
(This file will not be graded).
"""

import numpy as np
import skimage
from skimage import color


def preprocess_data(data, process_method='default'):
    """Preprocesses dataset.

    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1] by dividing by 255.
          2. Remove dataset mean. Average the images across the batch dimension.
             This will result in a mean image of dimension (8,8,3).
          3. Flatten images, data['image'] is converted to dimension (N, 8*8*3)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Convert from rgb to gray then back to rgb. Use skimage
          3. Take the absolute value of the difference with the original image.
          4. Remove dataset mean. Average the absolute value differences across
             the batch dimension. This will result in a mean of dimension (8,8,3).
          5. Flatten images, data['image'] is converted to dimension (N, 8*8*3)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    if process_method == 'raw':

      #Normalize
      data['image'] = data['image'] / 255

      #De-mean
      data = remove_data_mean(data)

      #Reshape
      data['image'] = np.reshape( data['image'],(data['image'].shape[0],data['image'].shape[1]*data['image'].shape[2]*data['image'].shape[3]) )

    elif process_method == 'default':
      #Normalize
      data['image'] = data['image'] / 255

      #RGB->Gray->RGB
      Temp = np.array(color.gray2rgb(color.rgb2gray(data['image'])))

      #Abs Difference btw
      data['image'] = abs(data['image'] - Temp)

      #De-mean
      data = remove_data_mean(data)

      #Reshape
      data['image'] = np.reshape( data['image'],(data['image'].shape[0],data['image'].shape[1]*data['image'].shape[2]*data['image'].shape[3]) )

      #print('In data_tools->Xshape:',data['image'].shape)
      #print('In data_tools->Yshape:',data['label'].shape)


    elif process_method == 'custom':
        # Design your own feature!
              #Normalize
      data['image'] = data['image'] / 255

      #RGB->Gray->RGB
      Temp = data['image']
      for i in range(data['image'].shape[0]):
        Temp[i] = np.array(color.rgb2hsv(data['image'][i]))
      #Temp = np.array(color.rgb2hsv(data['image']))

      #Abs Difference btw
      data['image'] = Temp

      #De-mean
      data = remove_data_mean(data)

      #Reshape
      data['image'] = np.reshape( data['image'],(data['image'].shape[0],data['image'].shape[1]*data['image'].shape[2]*data['image'].shape[3]) )
    return data


def compute_image_mean(data):
    """ Computes mean image.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        image_mean(numpy.ndarray): Avaerage across the example dimension.
    """

    image_mean = np.mean(data['image'],axis=0)
    return image_mean


def remove_data_mean(data):
    """Removes data mean.

    Args:
        data(dict): Python dict loaded using io_tools.

    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    data['image'] = data['image'] - compute_image_mean(data)
    return data
