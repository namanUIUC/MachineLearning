"""Main function for binary classifier"""
import tensorflow as tf
import numpy as np
from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.01
max_iters = 300

def main(_):
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    A, T = read_dataset_tf('../data/trainset','indexing.txt')

    # Initialize model.
    ndim = A.shape[1]
    model = LogisticModel_TF(ndim-1)

    # Build TensorFlow training graph
    model.build_graph(learn_rate)

    # Train model via gradient descent.
    val = model.fit(T, A, max_iters)

    y_predict = []
    for i in range(val.shape[0]):
        if (val[i]<0.5):
            y_predict.append(0)
        else:
            y_predict.append(1)
    y_predict = np.array(y_predict)






if __name__ == '__main__':
    tf.app.run()
