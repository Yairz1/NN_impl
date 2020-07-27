from random import shuffle
import numpy as np
from numpy import array, zeros, ones, average, trace


class SGD:
    def __init__(self, batch_size, m):
        self.batch_size = batch_size
        self.m = m

    def create_batches(self):
        indices = list(range(self.m))
        shuffle(indices)
        set_of_batches = []
        num_of_batches = (self.m // self.batch_size) + 1
        for i in range(num_of_batches):
            set_of_batches.append(indices[i * self.batch_size:(i + 1) * self.batch_size])
        return set_of_batches

    def optimize(self, W0, X, C, F, grad_F, lr=0.1, momentum=0):
        m, n = W0.shape
        batches = self.create_batches()
        W_history = zeros((m * n, len(batches) + 1))
        W_history[:, 0] = W0.copy().reshape(-1)

        dw = 0  # first step without momentum
        for i, batch in enumerate(batches):
            current_W = W_history[:, i].reshape(m, n)
            g_sum = zeros((m, n))  # its a vector not scalar.
            for idxs in batch:
                sample = X[:, idxs].reshape(X.shape[0], 1)
                labels = C[:, idxs].reshape(C.shape[0], 1)
                g_sum += grad_F(sample, current_W, labels)
            g = g_sum / len(batch)  # average
            # lr = self.armijo_search(X[:, batch], F, current_W, C[:, batch], g, -g, maxIter=30)
            dw = momentum * dw - lr * g.reshape(-1)
            W_history[:, i + 1] = W_history[:, i] + dw
        return average(W_history, axis=1).reshape(m, n)

    def armijo_search(self, x, f, W, C, grad_f, d, maxIter):
        alpha = 1
        betta = 0.5
        c = 1 / 10000
        for _ in range(maxIter):
            # armijo condition
            if f(x, W + alpha * d, C) <= f(x, W, C) + c * alpha * trace(grad_f.T @ d):
                return alpha
            else:
                alpha = alpha * betta
        print("failed")
        return alpha
