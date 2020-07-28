import unittest
from NeuralNetwork import NeuralNetwork
from Utils import data_factory
from activation import tanh
from numpy.random import randn

from loss_function import objective_soft_max


class test_neural_networks(unittest.TestCase):

    def test00_forward_tanh_sanity(self):
        C_train, C_val, X_train, X_val = data_factory('Swiss')  # options: 'Swiss','PeaksData','GMMData'
        n = X_train.shape[0]
        l = C_train.shape[0]
        f = lambda X, W: W.T @ X  # W.columns = X.rows, W.rows = X.rows
        net = NeuralNetwork(10, f=f, sigma=tanh, layer_dim=(n, n), output_dim=(n, l))
        prediction = net.forward(X_val)
        objective_value = objective_soft_max(X=None, W=None, C=C_val, WT_X=prediction)
        print(objective_value)
