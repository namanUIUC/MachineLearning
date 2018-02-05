"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


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


def train_model(processed_dataset, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        processed_dataset(list): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choise.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """

    if (shuffle):
        data = shuffle_data(processed_dataset)
    else:
        data = processed_dataset
        
    batch = make_batches(data, batch_size)
    
    total_batches = int(np.ceil(data[0].shape[0] / batch_size))
    epoch = 1
    batch_in_progress = 0

    for i in range(num_steps):
        x_batch = batch[batch_in_progress][0]
        y_batch = batch[batch_in_progress][1].reshape((batch[batch_in_progress][1].shape[0],1))
        
        update_step(x_batch,y_batch,model,learning_rate)

        batch_in_progress += 1
        
        if (batch_in_progress == total_batches):
            if (shuffle):
                data = shuffle_data(processed_dataset)
                batch = make_batches(data, batch_size)
                
            epoch += 1
            batch_in_progress = 0
    # print(model.w)



    """
    # Perform gradient descent.
    # pass
    for epoch in range(num_steps):
        if (shuffle):
            # data concatenation
            xtemp = processed_dataset[0] 
            ytemp = processed_dataset[1].reshape((processed_dataset[1].shape[0],1))  
            temp = np.concatenate((xtemp, ytemp), axis=1)  
        
            # data shuffle
            np.random.shuffle(temp)

            # data formating
            ynew = temp[:,-1:] 
            ynew = np.array(ynew) 
            ynew = ynew.astype(np.float64)  
            xnew = temp[:,:-1] 
            xnew = np.array(xnew) 
            xnew = xnew.astype(np.float64)  
            p = [xnew, ynew.reshape(ynew.shape[0])]

        else:
            p = processed_dataset
        
        # Batch creation
        containers = int(np.floor((p[0].shape[0]) / batch_size)) 
        remaining = (p[0].shape[0]) % batch_size 
        for i in range (0,containers):
            batch_x = p[0][i*batch_size:((i+1)*batch_size),:]
           
        # Last Batch
        if (remaining):
            batch_x = p[0][(i+1)*batch_size:(i+1)*batch_size+remaining]
           
        """

    return model


def update_step(x_batch, y_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).

    Args:
        x_batch(numpy.ndarray): input data of dimension (N, ndims).
        y_batch(numpy.ndarray): label data of dimension (N, 1).
        model(LinearModel): Initialized linear model.
    """
    model.w = model.w - learning_rate*model.backward(model.forward(x_batch),y_batch) 


def train_model_analytic(processed_dataset, model):
    """Computes and sets the optimal model weights (model.w).

    Args:
        processed_dataset(list): List of [x,y] processed
            from utils.data_tools.preprocess_data.
        model(LinearRegression): LinearRegression model.
    """
    xx = np.array(processed_dataset[0])
    yy = processed_dataset[1].reshape((processed_dataset[1].shape[0],1))
    # print(xx.shape,yy.shape)
    z = np.concatenate((xx,np.ones((xx.shape[0],1))),axis=1)
    # print(xx.shape, yy.shape, z.shape)
    # model.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(z),np.transpose(z))),np.transpose(z)),yy)
    zTz = np.matmul(np.transpose(z),z) + model.w_decay_factor*np.identity(z.shape[1])
    Inv_zTz = np.linalg.inv(zTz)
    Inv_zTz_zT = np.matmul(Inv_zTz,np.transpose(z))
    u =  np.matmul(Inv_zTz_zT,yy)
    model.w = u.reshape((u.shape[0],1))
    # print(model.w)
    
    


def eval_model(processed_dataset, model):
    """Performs evaluation on a dataset.

    Args:
        processed_dataset(list): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    x = processed_dataset[0]
    y = processed_dataset[1].reshape((processed_dataset[1].shape[0],1))

    f = model.forward(x)
    loss = model.total_loss(f,y)
    acc = 1 - np.mean(np.abs((f-y)/y))
    print(acc)
    return loss
