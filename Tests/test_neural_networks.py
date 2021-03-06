import unittest
from NeuralNetwork import NeuralNetwork
from Utils import data_factory
from function import tanh, ReLU_F, f_grad_W_mul_V, f_grad_X_mul_V, ReLU
from function import Function
from numpy.random import randn

from loss_function import objective_soft_max, objective_soft_max_gradient_W, objective_soft_max_gradient_X


class test_neural_networks(unittest.TestCase):

    def test00_forward_tanh_sanity(self):
        C_train, C_val, X_train, X_val = data_factory('Swiss')  # options: 'Swiss','PeaksData','GMMData'
        n = X_train.shape[0]
        l = C_train.shape[0]
        layer_function = Function(ReLU_F, f_grad_X_mul_V, f_grad_W_mul_V)
        model = NeuralNetwork(10, f=layer_function, sigma=tanh, layer_dim=(n, n), output_dim=(n, l))
        output = model(X_val)
        objective_value = objective_soft_max(X=None, W=None, C=C_val, WT_X=output)
        self.assertTrue(True)

    def test01_backward_tanh_sanity(self):
        C_train, C_val, X_train, X_val = data_factory('Swiss')  # options: 'Swiss','PeaksData','GMMData'
        n = X_train.shape[0]
        l = C_train.shape[0]
        layer_function = Function(ReLU_F, f_grad_X_mul_V, f_grad_W_mul_V)
        layer_num = 10
        model = NeuralNetwork(layer_num, f=layer_function, sigma=ReLU, layer_dim=(n, n), output_dim=(n, l))
        output = model(X_val)
        W = model.params[layer_num - 1]
        X = model.layers_inputs[layer_num - 1]
        gradients = model.backward(loss_gradient_x=objective_soft_max_gradient_X(X, W, C=C_val),
                                   loss_gradient_w=objective_soft_max_gradient_W(X, W, C=C_val))
        vec_grad = model.params_to_vector()
        self.assertTrue(True)
