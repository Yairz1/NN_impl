from matplotlib.pyplot import plot
from numpy import average, zeros, argmax
from numpy.random import randn

from Utils import data_factory
from loss_function import objective_soft_max, objective_soft_max_gradient_W
from optimizer import SGD


def accuracy(X, W, C):
    XT_W = X.T @ W
    res = argmax(XT_W, axis=1)
    expected = argmax(C, axis=0)
    diff = res - expected
    return round(100 * len(diff[diff == 0]) / len(diff), 4)


def train(C_train, C_val, X_train, X_val, batch_size, epochs, lr, momentum=0):
    # ----------------- hyper params init -----------------
    W0 = randn(X_train.shape[0], C_train.shape[0])
    m, n = W0.shape
    W = W0.copy()
    optimizer = SGD(batch_size=batch_size, m=X_train.shape[1])
    # ----------------------------------------------------

    # ----------------- stats lists init -----------------
    W_history = zeros((W.shape[0] * W.shape[1], epochs))
    val_score = []
    train_score = []
    train_acc = []
    val_acc = []
    # ----------------------------------------------------

    for epoch in range(epochs):
        W = optimizer.optimize(W, X_train, C_train,
                               objective_soft_max,
                               objective_soft_max_gradient_W, lr=lr, momentum=momentum)

        W_history[:, epoch] = W.reshape(W.shape[0] * W.shape[1])
        train_score.append(objective_soft_max(X_train, W, C_train))
        val_score.append(objective_soft_max(X_val, W, C_val))
        train_acc.append(accuracy(X_train, W, C_train))
        val_acc.append(accuracy(X_val, W, C_val))

    W_res = average(W_history, axis=1).reshape(m, n)
    train_score.append(objective_soft_max(X_train, W_res, C_train))
    val_score.append(objective_soft_max(X_val, W_res, C_val))
    # todo add plot epoch \ accuracy (wrote in train)
    plot(range(len(train_score)), train_score)
    return train_score, train_acc, val_score, val_acc
