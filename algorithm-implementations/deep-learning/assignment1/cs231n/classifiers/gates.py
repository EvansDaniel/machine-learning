import numpy as np

class MultiplyGate:
    def forward(self,W, X):
        return np.dot(X, W)

    def backward(self, W, act_input, dZ):
        dW = np.dot(np.transpose(act_input), dZ)
        dact_input = np.dot(dZ, np.transpose(W))
        return dW, dact_input

class AddGate:
    def forward(self, mul, b):
        return mul + b

    def backward(self, Xmul, dZ):
        dXmul = dZ * np.ones_like(Xmul) # ?
        db = np.sum(dZ, axis=0) # ?
        return db, dXmul

class ReLU:
    def forward(self, X):
        return np.maximum(0, X)
    
    def backward(self, indexer, X):
        X[indexer < 0] = 0
        return X
    
class Softmax:
    def forward(self, scores):
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
        f_y_i = - scores[np.arange(len(y)), y]
        norm = np.sum(np.exp(scores), axis=1)
        log_norm = np.log(norm)
        loss = np.mean(f_y_i + log_norm)
        # average cross-entropy loss and regularization
        for w in W:
            loss += 0.5 * reg * np.sum(w * w)
        return loss

    def derivative(self, probs, y):
        '''
        probs - output from softmax function
        y - labels, shape = (N,)
        '''
        num_examples = probs.shape[0]
        probs[range(num_examples), y] -= 1
        probs /= num_examples
        return probs