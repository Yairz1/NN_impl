from random import shuffle
import numpy as np
from numpy import array, zeros, ones, average


class SGD:
    def __init__(self, batch_size, m):
        self.batch_size = batch_size
        self.m = m

    def optimize(self, W0, X, C, F, grad_F, lr=0.001, momentum=0):
        W = W0.copy()
        m, n = W0.shape
        W = W.reshape(-1)
        batches = self.create_batches()
        W_history = zeros((W.shape[0], len(batches) + 1))
        W_history[:, 0] = W
        dw = 0  # first step without momentum
        for i, batch in enumerate(batches):
            grad_sum = 0
            for idxs in batch:
                sample = X[:, idxs].reshape(X.shape[0], 1)
                labels = C[:, idxs].reshape(C.shape[0], 1)
                grad_sum += grad_F(sample, W.reshape(m, n), labels)
            grad = grad_sum / len(batch)
            # lr = self.armijo_search(X[:, batch], F, grad, -grad, maxIter=30)
            dw = momentum * dw - lr * grad.reshape(-1)
            W_history[:, i + 1] = W_history[:, i] + dw
        return average(W_history, axis=1).reshape(m, n)  # 0 it's columns

    def create_batches(self):
        indices = list(range(self.m))
        shuffle(indices)
        set_of_batches = []
        num_of_batches = (self.m // self.batch_size) + 1
        for i in range(num_of_batches):
            set_of_batches.append(indices[i * self.batch_size:(i + 1) * self.batch_size])
        return set_of_batches

    def armijo_search(self, x, f, grad_f, d, maxIter):
        alpha = 1
        betta = 0.5
        c = 1 / 10000
        for _ in range(maxIter):
            # armijo condition
            if f(x + alpha * d) <= f(x) + c * alpha * (grad_f.T @ d):
                return alpha
            else:
                alpha = alpha * betta
        print("failed")
        return alpha
