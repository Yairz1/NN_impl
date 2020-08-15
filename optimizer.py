from random import shuffle
import numpy as np
from numpy import array, zeros, ones, average, trace
from numpy.linalg import norm

from loss_function import objective_soft_max_gradient_X, objective_soft_max_gradient_W, objective_soft_max


class SGD:
    def __init__(self, batch_size, m):
        self.batch_size = batch_size
        self.m = m

    def create_batches(self):
        indices = list(range(self.m))
        shuffle(indices)
        set_of_batches = []
        num_of_batches = (self.m // self.batch_size)
        for i in range(num_of_batches):
            set_of_batches.append(indices[i * self.batch_size:(i + 1) * self.batch_size])
        return set_of_batches

    # def optimize_origin(self, W0, X, C, F, grad_F, lr=0.1, momentum=0):
    # m, n = W0.shape
    # batches = self.create_batches()
    # W_history = zeros((m * n, len(batches) + 1))
    # W_history[:, 0] = W0.copy().reshape(-1)
    #
    # dw = 0  # first step without momentum
    # for i, batch in enumerate(batches):
    #     current_W = W_history[:, i].reshape(m, n)
    #     g_sum = zeros((m, n))  # its a vector not scalar.
    #     for idxs in batch:
    #         sample = X[:, idxs].reshape(X.shape[0], 1)
    #         labels = C[:, idxs].reshape(C.shape[0], 1)
    #         g_sum += grad_F(sample, current_W, labels)
    #     g = g_sum / len(batch)  # average
    #     lr = self.armijo_search(X[:, batch], F, current_W, C[:, batch], g, -g, maxIter=30)
    #     dw = momentum * dw - lr * g.reshape(-1)
    #     W_history[:, i + 1] = W_history[:, i] + dw
    # return average(W_history, axis=1).reshape(m, n)

    def optimize(self, W0, X, C, F, grad_F, lr=0.1, momentum=0, isNeural=False):
        m, n = W0.shape
        batches = self.create_batches()
        W_history = zeros((m * n, len(batches) + 1))
        W_history[:, 0] = W0.copy().reshape(-1)

        dw = 0  # first step without momentum
        for i, batch in enumerate(batches):
            if not isNeural:
                current_W = W_history[:, i].reshape(m, n)

                g = grad_F(X[:, batch], current_W, C[:, batch])
                # lr = self.armijo_search(X[:, batch], F, current_W, C[:, batch], g, -g, maxIter=30, isNeural=isNeural)

            else:
                output = F(X[:, batch])  # forward pass
                g = F.backward(
                    objective_soft_max_gradient_X(X=None, W=F.params[F.num_layers - 1][0], C=C[:, batch], WT_X=output),
                    objective_soft_max_gradient_W(X=F.layers_inputs[F.num_layers - 1], W=None, C=C[:, batch],
                                                  WT_X=output))
                W = F.params_to_vector(F.params).copy()

                lr = self.armijo_search(x=X[:, batch], f=F, W=W, C=C[:, batch], grad_f=g, d=-g, maxIter=100,
                                        isNeural=isNeural, momentum=(momentum,dw))
            dw = momentum * dw + lr * g.reshape(-1)
            W_history[:, i + 1] = W_history[:, i] - dw
        return W_history[:, i + 1].reshape(m, n)  # average(W_history, axis=1).reshape(m, n)

    def armijo_search(self, x, f, W, C, grad_f, d, maxIter, isNeural, momentum=0):
        alpha = 0.1
        betta = 0.5
        c = 1 / 10000
        if momentum != 0:
            (momentum, dw) = momentum
        if isNeural:
            objective1 = objective_soft_max(x, W=None, C=C, WT_X=f(x))
        for _ in range(maxIter):
            # armijo condition
            if not isNeural:
                isGood_alpha = f(x, W + alpha * d, C) <= f(x, W, C) + c * alpha * trace(grad_f.T @ d)
            else:
                if momentum != 0:
                    d = -(momentum * dw + alpha * d.reshape(-1)).reshape(-1, 1)
                f.set_params(W + alpha * d)
                objective2 = objective_soft_max(x, W=None, C=C, WT_X=f(x))
                isGood_alpha = objective2 <= objective1 + c * alpha * trace(grad_f.T @ d)

            if isGood_alpha:
                return alpha
            else:
                alpha = alpha * betta
        print("failed")
        return alpha
