from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class MultiplyGate:
    def forward(self,W, X):
        return np.dot(X, W)

    def backward(self, W, X, dZ):
        dW = np.dot(np.transpose(X), dZ)
        dX = np.dot(dZ, np.transpose(W))
        return dW, dX

class AddGate:
    def forward(self, X, b):
        return X + b

    def backward(self, X, b, dZ):
        dX = dZ * np.ones_like(X)
        db = np.dot(np.ones((1, dZ.shape[0]), dtype=np.float64), dZ)
        return db, dX
# Need dL/da1 = dL/dscores * dscores/da1
# dscores/da1
# dscores_da1 = W2
# da1 = dscores.dot(dscores_da1.T)
    
# Need dL/z1 = dL/da1 * da1/dz1 - This is the derivative of ReLU
# dz1 = da1
# dz1[a1 <= 0] = 0
class ReLU:
    def forward(self, X):
        return np.maximum(0, X)
    
    def backward(self, indexer, X):
        X[indexer < 0] = 0
        return X
    
class Softmax:
    def predict(self, scores):
        '''
        scores: output from last layer of network, shape = (N, C)
        '''
        scores -= np.max(scores, axis=1)[:, np.newaxis]
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
        return probs

class CrossEntropyLoss:
    # Cross entropy loss
    def loss(self, scores, y, W, reg):
        '''
        scores - output from last layer of network, shape = (N, C)
        y - labels, shape = (N,)
        '''
        # Take only correct classes  
        f_y_i = - scores[np.arange(N), y]
        norm = np.sum(np.exp(scores), axis=1)
        log_norm = np.log(norm)
        loss = np.mean(f_y_i + log_norm)
        W_sum = 0
        for w in W:
            W_sum += np.sum(w * w)
        loss += 0.5 * reg * W_sum
        return loss

    def diff(self, scores, y):
        '''
        scores - output from last layer of network, shape = (N, C)
        y - labels, shape = (N,)
        '''
        num_examples = scores.shape[0]
        probs = self.predict(scores)
        probs[range(num_examples), y] -= 1
        return probs

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """
  # layer_dims must be at least length three
  # for the input dims, 1 hidden layer, and class dims
  def __init__(self, layer_dims, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.W = []
    self.b = []
    for i in range(len(layer_dims) - 1):
        self.W.append(std * np.random.randn(layer_dims[i], layer_dims[i+1]))
        self.b.append(np.zeros(layer_dims[i+1]))
    
    #self.params = {}
    #self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    #self.params['b1'] = np.zeros(hidden_size)
    #self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    #self.params['b2'] = np.zeros(output_size)

  def propogate(self, X, y=None, reg=0.0, lr=1e-7):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    #W1, b1 = self.params['W1'], self.params['b1']
    #W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    
    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    
    # Initialize gates, add parameter to pass activation function
    # gate in 
    addGate = AddGate()
    multiplyGate = MultipyGate()
    activationGate = ReLU()
    softmaxGate = Softmax()
    crossEntropy = CrossEntropyLoss()
    
    # Do the forward pass 
    layer_input = X
    # Contains output from multiply gate, add gate, and layer_input
    forward_propagation = [None, None, layer_input]
    for i in range(len(self.W)):
        multipyOut = multiplyGate.forward(W[i], layer_input)
        addOut = addGate.forward(multipyOut, b[i])
        layer_input = activationGate.forward(addOut)
        forward.append((multipyOut, addOut, layer_input))
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    
    # compute the class probabilities
    #scores -= np.max(scores, axis=1)[:, np.newaxis]
    #exp_scores = np.exp(scores)
    #probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
    
    # Take only correct classes  
    #f_y_i = - scores[np.arange(N), y]
    #norm = np.sum(np.exp(scores), axis=1)
    #log_norm = np.log(norm)
    #loss = np.mean(f_y_i + log_norm)
    #loss += 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
    loss = crossEntropy.loss(layer_input, y, self.W, reg)
    
    # Backward pass
    forward_len = len(forward)
    # forward[forward_len - 1][2] is the last output from nodes before softmax
    dscores = crossEntropy.derivative(forward[forward_len - 1][2], y)
    for i in range(len(forward)-1, 0, -1):
        dadd  = activationGate.backward(
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    #grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    
    # dL/dscores i.e. derivative of cross entropy loss w.r.t its input 
    #dscores = probs
    #dscores[np.arange(N), y] -= 1
    #dscores /= N
    
    # Need dL/dW2 = dscores_dW2 * dL/dscores
    # dscores/dW2
    #dscores_dW2 = a1
    #dW2 = dscores_dW2.T.dot(dscores)
    # Add reg part of derivative
    #dW2 += reg * W2
    #grads['W2'] = dW2
    
    # Need dL/db2 = dscores/db2 * dL/dscores
    # dscores/db2
    #dscores_db2 = np.ones(dscores.shape[0])
    #db2 = dscores_db2.dot(dscores) # Same as np.sum(dscores, axis=0)
    #grads['b2'] = db2
    
    # Need dL/da1 = dL/dscores * dscores/da1
    # dscores/da1
    #dscores_da1 = W2
    #da1 = dscores.dot(dscores_da1.T)
    
    # Need dL/z1 = dL/da1 * da1/dz1 - This is the derivative of ReLU
    #dz1 = da1
    #dz1[a1 <= 0] = 0
    
    # Need dL/dW1 = dL/dz1 * dz1/dW1
    #dz1_dW1 = X
    #dW1 = dz1_dW1.T.dot(dz1)
    # reg part of derivative
    #dW1 += reg * W1 
    #grads['W1'] = dW1
    #print(W1.shape, dW1.shape, W2.shape, dW2.shape)
    
    # Need dL/db1 = dL/dz1 * dz1/db1
    #dscores_db1 = np.ones(dz1.shape[0])
    #db1 = dscores_db1.dot(dz1) # Same as np.sum(dz1, axis=0)
    #print(db1.shape)
    #grads['b1'] = db1
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      rand_indices = np.random.choice(np.arange(batch_size), batch_size)
      X_batch = X[rand_indices, :]
      y_batch = y[rand_indices]
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss = self.propogate(X_batch, y=y_batch, reg=reg, lr=learning_rate)
      #loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] += - learning_rate * grads['W1']
      self.params['b1'] += - learning_rate * grads['b1']
      self.params['W2'] += - learning_rate * grads['W2']
      self.params['b2'] += - learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    z1 = X.dot(self.params['W1']) + self.params['b1']
    a1 = np.maximum(0, z1)
    scores = a1.dot(self.params['W2']) + self.params['b2']
    
    # No need to compute softmax because similar to log function
    # it is increasing monotonically so the argmax of the input to 
    # the softmax is the argmax of its output
    y_pred = np.argmax(scores, axis=1)
    
    return y_pred