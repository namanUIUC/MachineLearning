import numpy as np
import scipy.io as scio

mnist = scio.loadmat('mnist_big.mat')

inp = mnist['X_train']/255.0

test = mnist['X_test']/255.0

inpD = []
for item in inp:
    inpD.append(item.reshape(28, 28))

testD = []
for item in test:
    testD.append(item.reshape(28,28))

mnist_2D = {}
mnist_2D['X_train'] = inpD
mnist_2D['Y_train'] = mnist['Y_train']
mnist_2D['X_test'] = testD
mnist_2D['Y_test'] = mnist['Y_test']

scio.savemat('mnist_2D.mat',mnist_2D)
