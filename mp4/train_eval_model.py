"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
import cvxopt
import cvxopt.solvers


def shuffle_data(pd):

     xtemp = pd[0]
     ytemp = pd[1].reshape((pd[1].shape[0],1))

     temp = np.concatenate((xtemp, ytemp), axis=1)

     np.random.shuffle(temp)

     ynew = temp[:,-1:]
     ynew = np.array(ynew)
     ynew = ynew.astype(np.float64)
     xnew = temp[:,:-1]
     xnew = np.array(xnew)
     xnew = xnew.astype(np.float64)

     p = [xnew, ynew.reshape(ynew.shape[0])]

     return p


def make_batches(pd,batch_size):
     containers = int(np.floor((pd[0].shape[0]) / batch_size))
     remaining = ((pd[0].shape[0]) % batch_size)
     batch = []
     for i in range (0,containers):
          batch_x = pd[0][i*batch_size:((i+1)*batch_size),:]
          batch_y = pd[1][i*batch_size:((i+1)*batch_size)]
          batch.append([batch_x, batch_y])
     if (remaining):
          batch_x = pd[0][(i+1)*batch_size:(i+1)*batch_size+remaining,:]
          batch_y = pd[1][(i+1)*batch_size:(i+1)*batch_size+remaining]
          batch.append([batch_x, batch_y])
     return batch


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.

    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.

    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.

    Returns:
        model(LinearModel): Returns a trained model.
    """

    # Performs gradient descent. (This function will not be graded.)
    processed_dataset = [data['image'], data['label']]
    if (shuffle):
        Tempdata = shuffle_data(processed_dataset)
    else:
        Tempdata = processed_dataset

    batch = make_batches(Tempdata, batch_size)

    total_batches = int(np.ceil(Tempdata[0].shape[0] / batch_size))
    epoch = 0
    batch_in_progress = 0

    for i in range(num_steps):
        x_batch = batch[batch_in_progress][0]
        y_batch = batch[batch_in_progress][1].reshape((batch[batch_in_progress][1].shape[0],1))

        update_step(x_batch,y_batch,model,learning_rate)

        batch_in_progress += 1

        if (batch_in_progress == total_batches):
            if (shuffle):
                Tempdata = shuffle_data(processed_dataset)
                batch = make_batches(Tempdata, batch_size)

            #print('epoch completed: ', epoch)
            #l,a = eval_model(data, model)
            #print('Loss: ',l)
            #print('Accuracy: ',a)
            epoch += 1
            batch_in_progress = 0

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    # Implementation here. (This function will not be graded.)
    grad = model.backward(model.forward(x_batch),y_batch)
    grad = np.reshape(grad,(grad.shape[0],1))
    model.w = model.w - learning_rate*grad


def train_model_qp(data, model):
    """Computes and sets the optimal model wegiths (model.w) using a QP solver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.
    """
    P, q, G, h = qp_helper(data, model)
    P = cvxopt.matrix(P, P.shape, 'd')
    q = cvxopt.matrix(q, q.shape, 'd')
    G = cvxopt.matrix(G, G.shape, 'd')
    h = cvxopt.matrix(h, h.shape, 'd')
    sol = cvxopt.solvers.qp(P, q, G, h)
    z = np.array(sol['x'])
    # Implementation here (do not modify the code above)

    # Set model.w
    try:
       n = data['image'].shape[1]
    except:
       n = data['image'].shape[0]
    model.w = z[:n+1]


def qp_helper(data, model):
    """Prepares arguments for the qpsolver.

    Args:
        data(dict): Data from utils.data_tools.preprocess_data.
        model(SupportVectorMachine): Support vector machine model.

    Returns:
        P(numpy.ndarray): P matrix in the qp program.
        q(numpy.ndarray): q matrix in the qp program.
        G(numpy.ndarray): G matrix in the qp program.
        h(numpy.ndarray): h matrix in the qp program.
    """
    P = None
    q = None
    G = None
    h = None
    # Implementation here.
    try:
        k = data['image'].shape[1]
        n = data['image'].shape[0]
    except:
        k = data['image'].shape[0]
        n = 1
    I = np.identity(k)
    size = k+1+n
    P = np.zeros((size,size))
    P[:I.shape[0],:I.shape[1]] = I
    P = model.w_decay_factor*P

    a = np.zeros((k+1,1))
    b = np.ones((n,1))
    q = np.concatenate((a,b),axis=0)

    # Making G
    x = data['image']
    x1 = np.hstack((x, np.ones((n,1))))
    y = data['label']
    x2 = np.multiply(-y, x1)
    G1 = np.hstack((x2, -np.eye(n)))

    g1 = np.zeros((n, k+1))
    g2 = -np.eye(n)
    G2 = np.hstack((g1,g2))

    G = np.vstack((G1,G2))

    h = -np.concatenate((np.ones((n,1)),np.zeros((n,1))),axis=0)
    #print(P.shape, q.shape, G.shape, h.shape)
    return P, q, G, h


def eval_model(data, model):
    """Performs evaluation on a dataset.

    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.

    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    # Implementation here.
    processed_dataset = [data['image'], data['label']]
    x = processed_dataset[0]
    y = processed_dataset[1].reshape((processed_dataset[1].shape[0],1))

    f = model.forward(x)
    f = f.astype(np.float64)
    loss = model.total_loss(f,y)

    y_predict = model.predict(f)
    # y_predict = np.reshape(y_predict,(y_predict.shape[0],1))
    assert(y.shape == y_predict.shape)
    # import pdb; pdb.set_trace()

    s = np.sum(y == y_predict)
    acc = 1. * s / len(y)
    return loss, acc
