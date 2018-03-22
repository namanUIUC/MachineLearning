import numpy as np
import scipy.io as scio
import sys
sys.path.append('../')
from convnet import cnn
import config
import signal
import sys

numEpoch = config.numEpoch
trainExamples = config.trainExamples
valExamples = config.valExamples
batchSize = config.batchSize
saveModel = config.saveModel
modelFile = config.modelFile
validate  = config.validate
pretrain = config.pretrain
trainedModel = config.trainedModel

log = config.log
trainlog = config.trainlog
vallog = config.vallog

net = cnn()

if pretrain:
    model = scio.loadmat(trainedModel)
    net.Weights = np.asarray(model['weights'][0])
    for i in range(5):
        net.Biases[i] = model['biases'][0][i][0]

mnist = scio.loadmat('../data/mnist_2D.mat')

def signal_handler(signal, frame):
            print('You pressed Ctrl+C!')
            if saveModel:
                SaveModel()
                print "Model Saved"
            sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def SaveModel():
    if saveModel:
        model= {}
        model['weights'] = net.Weights
        model['biases'] = net.Biases
        scio.savemat(modelFile, model)

if log:
    trlog = open( trainlog , 'w')
    vlog = open( vallog, 'w')
numIter = 1
for epoch in range(numEpoch):

    trainList = [np.random.randint(0,60000) for i in range(trainExamples)]
    valList = [np.random.randint(0,10000) for i in range(valExamples)]

    trainLabel = np.asarray([[0 for i in range(10)] for j in range(trainExamples)])
    trainData = np.zeros(( trainExamples, mnist['X_train'][0].shape[0], mnist['X_train'][0].shape[1] ))

    valLabel = np.asarray([[0 for i in range(10)] for j in range(valExamples)])
    valData = np.zeros(( valExamples, mnist['X_train'][0].shape[0], mnist['X_train'][0].shape[1] ))

    j=0
    for i in trainList:
        trainLabel[j,mnist['Y_train'][i]] = 1
        trainData[j] = mnist['X_train'][i]
        j += 1

    j=0
    for i in valList:
        valLabel[j,mnist['Y_test'][i]] = 1
        valData[j] = mnist['X_test'][i]
        j += 1


    j = 0
    while( j < trainExamples ):

        batchData = trainData[j:j+batchSize]
        batchLabel = trainLabel[j:j+batchSize]

        batchLoss = net.backward(batchData, batchLabel)

        print 'Iteration ',numIter, ': Train Loss =  ' ,batchLoss

        if log:
            trlog.write(str(numIter) + ' ' + str(batchLoss) + '\n')

        numIter += 1
        j += batchSize

    ### Debugging: For Overfitting exercise
    # acc = 0
    # for i in range(trainExamples):
        # if trainLabel[i][net.predict(trainData[i])] == 1:
            # acc += 1

    # print 'Epoch ', epoch+1, " : Accuracy:  ", acc*100.0/trainExamples

    ### Validation
    acc = 0
    val_loss = 0
    for i in range(valExamples):
        [predict,loss] = net.validate(valData[i], valLabel[i])

        if valLabel[i][predict] == 1:
            acc += 1

        val_loss += loss

    if validate:
        print 'Epoch ', epoch+1,"Validation Loss: ",val_loss/valExamples, ",Accuracy:  ", acc*100.0/valExamples
    if log:
        vlog.write(str(epoch+1)+ '  '+ str(val_loss/valExamples)+'  '+ str(acc*100.0/valExamples)+ '\n')

if log:
    trlog.close()
    vlog.close()

SaveModel()
