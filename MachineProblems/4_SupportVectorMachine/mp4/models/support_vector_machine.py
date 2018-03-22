"""Implements support vector machine."""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from models.linear_model import LinearModel


class SupportVectorMachine(LinearModel):
    """Implements a linear regression mode model"""

    def backward(self, f, y):
        """Performs the backward operation based on the loss in total_loss.

        By backward operation, it means to compute the gradient of the loss
        w.r.t w.

        Hint: You may need to use self.x, and you made need to change the
        forward operation.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_grad(numpy.ndarray): Gradient of L w.r.t to self.w,
              dimension (ndims+1,).
        """
        reg_grad = None
        loss_grad = None
        # Implementation here.
        reg_grad = self.w_decay_factor*self.w

        # f = np.reshape(f,(f.shape[0],1))
        # y = np.reshape(y,(y.shape[0],1))
        yx = (y*self.x)
        ind = np.zeros((y.shape[0],1))
        #print(f.shape,y.shape,yx.shape,ind.shape)
        ind[y*f <1 ] = 1
        loss_grad = - np.matmul(np.transpose(yx),ind)

        total_grad = reg_grad + loss_grad

        return total_grad.flatten()

    def total_loss(self, f, y):
        """The sum of the loss across batch examples + L2 regularization.
        Total loss is hinge_loss + w_decay_factor/2*||w||^2

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
            y(numpy.ndarray): Ground truth label, dimension (N,1).
        Returns:
            total_loss (float): sum hinge loss + reguarlization.
        """
        f = (np.reshape(f,(f.shape[0],1)))
        f = f.astype(np.float64)
        y = (np.reshape(y,(y.shape[0],1)))
        y = y.astype(np.float64)

        hinge_loss = np.sum(np.maximum(0, 1 - f*y))
        l2_loss = (0.5*self.w_decay_factor*(np.linalg.norm(self.w))**2)
        # Implementation here.
        #pass

        total_loss = hinge_loss + l2_loss
        #print(total_loss)
        return total_loss

    def predict(self, f):
        """Converts score to prediction.

        Args:
            f(numpy.ndarray): Output of forward operation, dimension (N,1).
        Returns:
            (numpy.ndarray): Hard predictions from the score, f,
              dimension (N,1). Tie break 0 to 1.0.
        """
        y_predict = []
        for i in range(f.shape[0]):
            if(f[i][0] < 0):
                y_predict.append(-1)
            else:
                y_predict.append(1)
        y_predict = np.array(y_predict).reshape(f.shape[0],1)

        return y_predict
