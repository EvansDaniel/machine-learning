import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  for i in range(num_train):
        s_i = X[i].dot(W) # Shape (c,)
        
        # Keep numerical stability 
        s_i -= np.max(s_i)
        
        norm = np.sum(np.exp(s_i))
        
        l_i = -s_i[y[i]] + np.log(norm)
        loss += l_i
        
        for j in range(num_classes):
            p_j = np.exp(s_i[j]) / np.float64(norm) # scalar
            dW[:, j] += X[i] * (p_j - (y[i] == j))
  
  loss /= np.float64(num_train)
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= np.float(num_train)
  dW += reg * W
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  # L_i= −f_y_i + log ∑_j (e^f_j)
  scores = X.dot(W) # shape (N, C)

  # Prevent numeric instability
  scores -= np.matrix(np.max(scores, axis=1)).T
  f_y_i = - scores[np.arange(num_train), y]
  norm = np.sum(np.exp(scores), axis=1)
  log_norm = np.log(norm)
  loss = np.mean(f_y_i + log_norm)
  loss += 0.5 * 9 * np.sum(W * W)

  # np.matrix(np.float64(norm)).T in R{num_train x 1}
  p = np.exp(scores) / np.matrix(np.float64(norm)).T # R{num_train x num_classes}
  # indicator function 
  p[np.arange(num_train), y] -= 1

  dW = X.T.dot(p)
  dW /= np.float64(num_train)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

