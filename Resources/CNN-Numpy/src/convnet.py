import numpy as np
from scipy import signal
import activations as act
import sys
sys.path.append('../')
import config
from fwd import *
from back import *

inp_width = config.width
inp_height = config.height
inp_channels = config.channels

layers = config.layers

initBias = config.initBias
activation = config.activation
lr = config.lr
alpha = config.alpha

class cnn:
    def __init__(self):

        n = inp_width*inp_height
        self.Weights = [np.random.randn(layers[0][1],inp_channels,layers[0][2],layers[0][2])/np.sqrt(n)]
        out_Size =  inp_width - layers[0][2] + 1 ########### Only for Height = Width
        self.Biases = [initBias*np.ones( layers[0][1] )]

        self.poolParams = [(layers[1][1], layers[1][2])]
        out_Size = out_Size/2  ########## Only for Kernel = 2 and Stride = 2

        n = out_Size*out_Size*layers[0][1]
        self.Weights.append(np.random.randn(layers[2][1],layers[0][1],layers[2][2],layers[2][2])/np.sqrt(n))
        out_Size = out_Size - layers[2][2]+1
        self.Biases.append(initBias*np.ones(layers[2][1]))

        self.poolParams.append((layers[3][1],layers[3][2]))
        out_Size = out_Size/2  ########## Only for Kernel = 2 and Stride = 2

        n = out_Size*out_Size*layers[2][1]
        self.Weights.append(np.random.randn(layers[4][1],layers[2][1],out_Size,out_Size)/np.sqrt(n))
        out_Size = 1
        self.Biases.append(initBias*np.ones(layers[4][1]))

        n = layers[4][1]
        self.Weights.append(np.random.randn(layers[5][1],layers[4][1])/np.sqrt(n))
        self.Biases.append(initBias*np.ones(layers[5][1]))

        n = layers[5][1]
        self.Weights.append(np.random.randn(layers[6][1],layers[5][1])/np.sqrt(n))
        self.Biases.append(initBias*np.ones(layers[6][1]))

        self.Weights = np.asarray(self.Weights)
        self.Biases = np.asarray(self.Biases)

        DirW = []
        DirB = []
        for i in range(5):
            DirW.append(np.zeros(self.Weights[i].shape))
            DirB.append(np.zeros(self.Biases[i].shape))
        self.DirW = np.asarray(DirW)
        self.DirB = np.asarray(DirB)

    def forward(self, inputData):

        weights = self.Weights
        biases = self.Biases
        poolParams = self.poolParams

        # layer0 = input Layer
        layer0 = np.asarray(inputData)

        # layer1 = conv1 layer
        layer1 = convFwd(np.asarray([layer0]),weights[0],biases[0])
        # layer2 = pool1 layer
        layer2 = poolFwd(layer1, poolParams[0][0], poolParams[0][1])
        # layer2 = convpool(np.asarray([layer0]),weights[0],biases[0], poolParams[0][0], poolParams[0][1])

        # layer3 = conv2 layer
        layer3 = convFwd(layer2,weights[1],biases[1])
        # layer4 = pool2 layer
        layer4 = poolFwd(layer3, poolParams[1][0], poolParams[1][1])
        # layer4 = convpool(layer2,weights[1],biases[1], poolParams[1][0], poolParams[1][1])

        # layer5 = fc1 layer
        layer5 = convFwd( layer4,weights[2] ,biases[2] )

        # layer6 = fc2 layer
        layer6 = act.activation(np.dot(weights[3],layer5[:,0]).transpose() + biases[3] , activation ).transpose()

        # layer7 = softmax layer
        layer7 = np.dot( weights[4], layer6[:,0] ).transpose() + biases[4]
        layer7 -= np.max(layer7)
        layer7 = np.exp(layer7)/sum(np.exp(layer7))

        return layer7

    def trloss(self, trainData, trainLabel):

        return -1*sum( trainLabel * np.log(self.forward(trainData)) )

    def predict(self, inputVal):

        outProb = self.forward(inputVal)
        return outProb.argmax()

    def validate(self, inputVal, inputLabel):

        outProb = self.forward(inputVal)
        loss = -1*sum( inputLabel * np.log(outProb) )

        return [outProb.argmax(),loss]

    def backward(self, trainData, trainLabel ):

        assert( len(trainData) == len(trainLabel)), 'Equal to Batch Size'

        batchSize = len(trainData)

        weights = self.Weights
        biases = self.Biases
        DirW = self.DirW
        DirB = self.DirB
        poolParams = self.poolParams

        # dWeights = np.zeros(weights.shape)
        # dBiases = np.zeros(biases.shape)
        dW4 = np.zeros(weights[4].shape)
        dB4 = np.zeros(biases[4].shape)

        dW3 = np.zeros(weights[3].shape)
        dB3 = np.zeros(biases[3].shape)

        dW2 = np.zeros(weights[2].shape)
        dB2 = np.zeros(biases[2].shape)

        dW1 = np.zeros(weights[1].shape)
        dB1 = np.zeros(biases[1].shape)

        dW0 = np.zeros(weights[0].shape)
        dB0 = np.zeros(biases[0].shape)

        loss = 0

        for image in range(batchSize):

            X_data = trainData[image]
            X_label = trainLabel[image]

            ###Forward Pass
            # layer0 = input Layer
            layer0 = np.asarray(X_data)

            # layer1 = conv1 layer
            layer1 = convFwd(np.asarray([layer0]),weights[0],biases[0])

            # layer2 = pool1 layer
            layer2 = poolFwd(layer1, poolParams[0][0], poolParams[0][1])

            # layer3 = conv2 layer
            layer3 = convFwd(layer2,weights[1],biases[1])

            # layer4 = pool2 layer
            layer4 = poolFwd(layer3, poolParams[1][0], poolParams[1][1])

            # layer5 = fc1 layer
            layer5 = convFwd( layer4,weights[2] ,biases[2] )

            # layer6 = fc2 layer
            layer6 = act.activation(np.dot(weights[3],layer5[:,0]).transpose() + biases[3] , activation ).transpose()

            # layer7 = softmax layer
            layer7 = np.dot( weights[4], layer6[:,0] ).transpose() + biases[4]
            layer7 -= np.max(layer7)
            layer7 = np.exp(layer7)/sum(np.exp(layer7))

            loss += -1*sum( X_label * np.log(layer7) )

            ### Gradients Accumulate
            dy = -1*(X_label - layer7)/2

            [dy, dW, dB ] = fcback(layer6, np.asarray([dy]).transpose() , weights[4])
            dW4 += dW
            dB4 += dB.flatten()
            dy = act.backActivate(dy.transpose(), layer6, activation)

            [dy, dW, dB ] = fcback(layer5[:,0], dy, weights[3])
            dW3 += dW
            dB3 += dB.flatten()
            dy = act.backActivate(dy.transpose(), layer5[:,0], activation)

            [dy, dW, dB ] = convBack(layer4, dy, weights[2])
            dW2 += dW
            dB2 += dB.flatten()

            dy = poolback(layer3, dy)
            dy = act.backActivate(dy, layer3, activation)

            [dy, dW, dB ] = convBack(layer2, dy, weights[1])
            dW1 += dW
            dB1 += dB.flatten()

            dy = poolback(layer1, dy)
            dy = act.backActivate(dy, layer1, activation)

            [dy, dW, dB ] = convBack(np.asarray([layer0]), dy, weights[0])
            dW0 += dW
            dB0 += dB.flatten()

        # Updates
        DirW[0] = alpha*DirW[0] - lr*dW0/batchSize
        weights[0] += DirW[0]

        DirW[1] = alpha*DirW[1] - lr*dW1/batchSize
        weights[1] += DirW[1]

        DirW[2] = alpha*DirW[2] - lr*dW2/batchSize
        weights[2] += DirW[2]

        DirW[3] = alpha*DirW[3] - lr*dW3/batchSize
        weights[3] += DirW[3]

        DirW[4] = alpha*DirW[4] - lr*dW4/batchSize
        weights[4] += DirW[4]

        DirB[0] = alpha*DirB[0] - lr*dB0/batchSize
        biases[0] += DirB[0]

        DirB[1] = alpha*DirB[1] - lr*dB1/batchSize
        biases[1] += DirB[1]

        DirB[2] = alpha*DirB[2] - lr*dB2/batchSize
        biases[2] += DirB[2]

        DirB[3] = alpha*DirB[3] - lr*dB3/batchSize
        biases[3] += DirB[3]

        DirB[4] = alpha*DirB[4] - lr*dB4/batchSize
        biases[4] += DirB[4]

        self.Weights = weights
        self.Biases = biases

        # return [loss/batchSize, dW4/batchSize]
        return loss/batchSize




