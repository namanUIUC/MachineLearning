from scipy import signal
import numpy as np
import activations as act
import sys
sys.path.append('../')
import config

def conv(X, convFilters):

    featureMaps = []
    for i in range(len(convFilters)):
        featureMap = []
        convFilter = convFilters[i]
        depth = len(convFilter)
        assert(depth == len(X)), 'Dimension Mismatch'
        for j in range(depth):
            featureMap.append(signal.convolve2d(X[j], np.rot90(convFilter[j],2),'valid'))

        featureMaps.append( sum(featureMap) )

    return np.asarray(featureMaps)


def convBack(X, dy, W):

    ### Compute dx, dW , dB

    if dy.shape[1] == 1:
        temp = dy
        dy = np.zeros((dy.shape[0],1,1))
        dy[:,0] = temp

    Wb = np.zeros((W.shape[1],W.shape[0],W.shape[2],W.shape[3]))
    dW = []
    dB = []

    for i in range(W.shape[0]):
        kernel = []
        for j in range(W.shape[1]):
            kernel.append( signal.convolve2d(X[j], np.rot90(dy[i],2) ,'valid') )
            Wb[j,i] = np.rot90(W[i,j],2)
        dW.append(np.asarray(kernel))
        dB.append(np.sum(dy[i]))

    pad = W.shape[2]-1
    dy = np.pad(dy, pad ,'constant' )[pad:-1*pad]

    dX = conv(dy, Wb)

    dW = np.asarray(dW)
    dB = np.asarray(dB)

    return [dX, dW, dB]

def fcback(X, dy, W):

    dX = np.dot( dy.transpose(), W )
    dW = np.dot( dy , X.transpose() )
    dB = dy

    return [ dX, dW, dB ]

def poolback(X, dy):

    ## To Improve
    dX = np.zeros(X.shape)

    for k in range(X.shape[0]):
        for i in range(0,X.shape[1],2):
            for j in range(0,X.shape[2],2):
                a =  X[k,i:i+2,j:j+2]
                ind = np.unravel_index(a.argmax(), a.shape)
                dX[k,i+ind[0],j+ind[1]] = dy[k,i/2,j/2]

    return dX

# def convBack(X, dy, W):

    ## Compute dx

    # dy = np.pad(dy, W.shape[2]-1 ,'constant' )[1:-1]

    # Wb = []
    # for j in range(W.shape[1]):
        # kernel = []
        # for i in range(W.shape[0]):
            # kernel.append( np.rot90(W[i,j],2) )
        # Wb.append(np.asarray(kernel))

    # dX = conv(dy, Wb)

    ## Compute dW

    # dW = []
    # for i in range(dy.shape[0]):
        # kernel = []
        # for j in range(X.shape[0]):
            # kernel.append( signal.convolve2d(X[j], dy[i],'valid') )
        # dW.append(np.asarray(kernel))

    # dW = np.asarray(dW)

    ## Compute dB

    # dB = []
    # for i in range(dy.shape[0]):
        # dB.append(sum(dy[i]))
    # dB = np.asarray(dB)

    ## Return dX, dW, dB

    # return [dX, dW, dB]
