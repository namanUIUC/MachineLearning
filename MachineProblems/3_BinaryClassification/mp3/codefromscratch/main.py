"""Main function for binary classifier"""

import numpy as np

from io_tools import *
from logistic_model import *

""" Hyperparameter for Training """
learn_rate = 0.0005
max_iters = 300

if __name__ == '__main__':
    ###############################################################
    # Fill your code in this function to learn the general flow
    # (..., although this funciton will not be graded)
    ###############################################################

    # Load dataset.
    # Hint: A, T = read_dataset('../data/trainset','indexing.txt')
    A, T = read_dataset('../data/trainset','indexing.txt')

    #x_train, x_test, y_train, y_test = train_test_split(A, T, test_size=0.2)

    # Initialize model.
    ndim = (A.shape[1])-1
    model = LogisticModel(ndim, 'ones')

    #print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # Train model via gradient descent.
    model.fit(T, A, learn_rate, max_iters)

    # Save trained model to 'trained_weights.np'
    model.save_model('trained_weights.np')

    # Load trained model from 'trained_weights.np'
    model.load_model('trained_weights.np')

    # Try all other methods: forward, backward, classify, compute accuracy
    y_predict = model.classify(A)

    # s = np.sum(T == y_predict)
    # acc = s / len(T)
    # print('Final Accuracy : ', acc)



