import numpy as np

def activation(x, func):

    if func == 'relu':

        return (x+abs(x))/2
    elif func == 'tanh':

        return ( ( 1-np.exp( -2*x ) )/( 1 + np.exp( -2*x ) ) )
    else :
        assert(0 == 1), 'Invalid Activation Function'

# def backActivate(error, inUnits, outUnits,  func):
def backActivate(error, inUnits,  func):

    if func == 'relu':

        def ReLU(y):
            return 1 if y > 0 else 0

        ReLU = np.vectorize(ReLU)

        return error*ReLU(inUnits)
    # elif func == 'tanh':

        # def tanh(y):
            # return 1-y*y
        # tanh = np.vectorize(tanh)

        # return error*tanh(outUnits)
    else :
        assert(0 == 1), 'Invalid Activation Function'



