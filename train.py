from matplotlib.pyplot import plot
from numpy import average, zeros, argmax, sqrt
from numpy.random import randn

from Utils import data_factory
from loss_function import objective_soft_max, objective_soft_max_gradient_W
from optimizer import SGD
from scipy.special import softmax


def accuracy(X, W, C, XT_W=None):
    if XT_W is None:
        XT_W = X.T @ W
        res = argmax(XT_W, axis=1)
    else:
        res = argmax(softmax(XT_W, axis=1), axis=0)

    expected = argmax(C, axis=0)
    diff = res - expected
    return round(100 * len(diff[diff == 0]) / len(diff), 2)


def train(model, model_grad, isNeural, C_train, C_val, X_train, X_val, W0, batch_size, epochs, lr, momentum=0,
          patient=100):
    # ----------------- hyper params init -----------------
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
        # if epoch % patient == 0:
        #     lr = lr / 10
        W = optimizer.optimize(W0=W, X=X_train,
                               C=C_train,
                               F=model,
                               grad_F=model_grad,
                               lr=lr,
                               momentum=momentum,
                               isNeural=isNeural)
        # lr = lr /(sqrt(epoch+1))
        W_history[:, epoch] = W.reshape(W.shape[0] * W.shape[1])
        if not isNeural:
            train_score.append(objective_soft_max(X_train, W, C_train))
            val_score.append(objective_soft_max(X_val, W, C_val))
            train_acc.append(accuracy(X_train, W, C_train))
            val_acc.append(accuracy(X_val, W, C_val))
        else:
            model.set_params(W_history[:, epoch])
            train_output = model(X_train)
            val_output = model(X_val)
            train_score.append(objective_soft_max(X_train, W=None, C=C_train, WT_X=train_output))
            val_score.append(objective_soft_max(X_val, W=None, C=C_val, WT_X=val_output))
            t_acc = accuracy(X_train, W, C_train, XT_W=train_output)
            train_acc.append(t_acc)
            v_acc = accuracy(X_val, W, C_val, XT_W=val_output)
            val_acc.append(v_acc)
            print(f'epoch : {epoch} train accuracy: {t_acc} val accuracy: {v_acc}')

    if not isNeural:
        W_res = average(W_history, axis=1).reshape(m, n)
        train_score.append(objective_soft_max(X_train, W_res, C_train))
        val_score.append(objective_soft_max(X_val, W_res, C_val))
    else:
        W_res = average(W_history, axis=1)
        model.set_params(W_res)
        train_output = model(X_train)
        val_output = model(X_val)
        train_score.append(objective_soft_max(X_train, W=None, C=C_train, WT_X=train_output))
        val_score.append(objective_soft_max(X_val, W=None, C=C_val, WT_X=val_output))
    plot(range(len(train_score)), train_score)
    return train_score, train_acc, val_score, val_acc
