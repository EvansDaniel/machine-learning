from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.gates import *

class TwoLayerNet:
    def __init__(self, layer_dims, std=1e-4):
        self.W = []
        self.b = []
        for i in range(len(layer_dims) - 1):
            self.W.append(std * np.random.randn(layer_dims[i], layer_dims[i+1]))
            self.b.append(np.zeros(layer_dims[i+1]))
        
        self.params = {} 
        self.params['W1'] = self.W[0] 
        self.params['b1'] = self.b[0] 
        self.params['W2'] = self.W[1]
        self.params['b2'] = self.b[1]

    def loss(self, X, y=None, reg=0.0, lr=1e-7):
        N, D = X.shape
        scores = None

        # Initialize gates, add parameter to pass activation function
        # gate in 
        addGate = AddGate()
        multiplyGate = MultiplyGate()
        activationGate = ReLU()
        softmaxGate = Softmax()
        crossEntropy = CrossEntropyLoss()
        
        actOut = X
        multipyOut = multiplyGate.forward(self.W[0], actOut)
        addOut = addGate.forward(multipyOut, self.b[0])
        forward = [(actOut, addOut, multipyOut)]
        print(len(self.W))
        for i in range(1, len(self.W)):
            actOut = activationGate.forward(addOut)
            multipyOut = multiplyGate.forward(self.W[i], actOut)
            addOut = addGate.forward(multipyOut, self.b[i])
            forward.append((actOut, multipyOut, addOut))
        
        scores = addOut
        
        # If the targets are not given then jump out, we're done
        if y is None:
          return scores

        # Compute the loss
        loss = crossEntropy.loss(scores, y, self.W, reg)
        
        #################################################################################

        probs = softmaxGate.forward(scores)
        
        # Backward pass
        forward_len = len(forward)
        # forward[forward_len - 1][2] is last addGate
        dadd = crossEntropy.derivative(probs
                                          , y)
        grad = []
        # forward_propgation[i] = [actOut, multipyOut, addOut]
        for i in range(forward_len-1, -1, -1):
            db, dmult = addGate.backward(forward[i][1], 
                                         dadd)
            dW, dRelu = multiplyGate.backward(self.W[i], 
                                             forward[i][0], 
                                             dmult)
            if i >= 1:
                dadd = activationGate.backward(forward[i-1][2], dRelu)
            dW += reg * self.W[i]
            grad.append((dW, db))
        
        grad = list(reversed(grad))
        #params = {} #params['W1'] = grad[1][0] #params['b1'] = np.array(grad[1][1]) #params['W2'] = grad[0][0]
        #params['b2'] = np.array(grad[0][1])
        return loss, grad

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
          X_batch = None
          y_batch = None

          rand_indices = np.random.choice(np.arange(batch_size), batch_size)
          X_batch = X[rand_indices, :]
          y_batch = y[rand_indices]

          # Compute loss and gradients using the current minibatch
          loss, grads = self.loss(X_batch, y=y_batch, reg=reg, lr=learning_rate)
          #loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
          loss_history.append(loss)

          for i in range(len(grads)):
            self.W[i] += - learning_rate * grads[i][0]
            self.b[i] += - learning_rate * np.array(grads[i][1])
          #self.params['W1'] += - learning_rate * grads['W1'] #self.params['b1'] += - learning_rate * grads['b1']
          #self.params['W2'] += - learning_rate * grads['W2'] #self.params['b2'] += - learning_rate * grads['b2']

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
        y_pred = None
        
        addGate = AddGate()
        multiplyGate = MultiplyGate()
        activationGate = ReLU()
        softmaxGate = Softmax()
        crossEntropy = CrossEntropyLoss()
        
        actOut = X
        multipyOut = multiplyGate.forward(self.W[0], actOut)
        addOut = addGate.forward(multipyOut, self.b[0])
        forward_propagation = [(actOut, addOut, multipyOut)]
        for i in range(1, len(self.W)):
            actOut = activationGate.forward(addOut)
            multipyOut = multiplyGate.forward(self.W[i], actOut)
            addOut = addGate.forward(multipyOut, self.b[i])
            forward_propagation.append((actOut, multipyOut, addOut))
        
        scores = addOut

        #z1 = X.dot(self.params['W1']) + self.params['b1']
        #a1 = np.maximum(0, z1)
        #scores = a1.dot(self.params['W2']) + self.params['b2']

        # No need to compute softmax because similar to log function
        # it is increasing monotonically so the argmax of the input to 
        # the softmax is the argmax of its output
        y_pred = np.argmax(scores, axis=1)

        return y_pred