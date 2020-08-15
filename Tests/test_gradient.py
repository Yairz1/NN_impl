import unittest

from NeuralNetwork import NeuralNetwork
from Utils import create_C_W_X_d, data_factory, show_and_save_plot, show_and_save_plot2
from numpy import average, round, trace, array
from numpy.linalg import norm
from numpy.random import randn, rand
from function import *
from loss_function import objective_soft_max, objective_soft_max_gradient_W, objective_soft_max_gradient_X


class test_gradient(unittest.TestCase):
    # ________________ loss tests __________________
    def test00_linear_convergence_loss_gradient_test_W(self):
        max_iter = 25
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, W, X, d, _ = create_C_W_X_d()
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(abs(objective_soft_max(X, W + eps * d, C) -
                              objective_soft_max(X, W, C)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.40 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), loss_convergence, range(len(loss_convergence)), loss_convergence,
                           title=f'Gradient_test_Linar_X', semilog=False, label1='Linar', label2='Quadratic', two=False , save_option=False, y_axis=False)


    def test01_linear_convergence_loss_gradient_test_X(self):
        max_iter = 17
        C, W, X, _, d = create_C_W_X_d()
        eps0 = 0.5
        losses = []
        loss_convergence = []
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = round(objective_soft_max(X + eps * d, W, C), 10)
            objective2 = round(objective_soft_max(X, W, C), 10)
            losses.append(abs(objective1 - objective2))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.4 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')


        show_and_save_plot(range(len(loss_convergence)), loss_convergence, range(len(loss_convergence)), loss_convergence,
                           title=f'Gradient_test_Linar_W', semilog=False, label1='Linar', label2='Quadratic', two=False , save_option=False)

    def test02_quadratic_convergence_loss_gradient_W(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, W, X, d, _ = create_C_W_X_d()
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = objective_soft_max(X, W + eps * d, C)
            objective2 = objective_soft_max(X, W, C)
            objective3 = eps * trace(d.T @ objective_soft_max_gradient_W(X, W, C))

            losses.append(abs(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')


        show_and_save_plot(range(len(loss_convergence)), round(loss_convergence,2), range(len(loss_convergence)), loss_convergence,
                           title=f'Gradient_test_Quadratic_W', semilog=False, label1='Quadratic', label2='Linar', two=False , save_option=False)


        #show_and_save_plot2(range(len(loss_convergence)), round(loss_convergence,2), title=f'Gradient_test_Quadratic_W')


    def test03_quadratic_convergence_loss_gradient_X(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, W, X, _, d = create_C_W_X_d()
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = objective_soft_max(X + eps * d, W, C)
            objective2 = objective_soft_max(X, W, C)
            objective3 = eps * trace(d.T @ objective_soft_max_gradient_X(X, W, C))

            losses.append(abs(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), round(loss_convergence,2), range(len(loss_convergence)), loss_convergence,
                           title=f'Gradient_test_Quadratic_X', semilog=False, label1='Quadratic', label2='Linar', two=False , save_option=False)

    # ________________ Specific layer linear tests __________________

    def test04_linear_convergence_gradient_test_W(self):
        max_iter = 25
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(X.shape[0], X.shape[0])
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(norm(layer_function(X, W + eps * d, b) -
                               layer_function(X, W, b)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.40 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), round(loss_convergence,2), range(len(loss_convergence)), loss_convergence,
                           title=f'linear_layer_convergence_gradient_test_W', semilog=False, label1='Linar', label2='Quadratic', two=False , save_option=False)

    def test05_linear_convergence_gradient_test_X(self):
        max_iter = 25
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(*X.shape)
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(norm(layer_function(X + eps * d, W, b) -
                               layer_function(X, W, b)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.40 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), round(loss_convergence,2), range(len(loss_convergence)), loss_convergence,
                           title=f'linear_layer_convergence_gradient_test_X', semilog=False, label1='Linar', label2='Quadratic', two=False , save_option=False)

    def test06_linear_convergence_gradient_test_b(self):
        max_iter = 25
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(*b.shape)
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            losses.append(norm(layer_function(X, W, b + eps * d) -
                               layer_function(X, W, b)))

        for i in range(1, len(losses)):
            loss_convergence.append(losses[i] / losses[i - 1])
        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.40 <= avg_val <= 0.6, msg=f'avg value = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), round(loss_convergence,2), range(len(loss_convergence)), loss_convergence,
                           title=f'linear_layer_convergence_gradient_test_b', semilog=False, label1='Linar', label2='Quadratic', two=False , save_option=False)

    # ___________________________________________________________
    # ________________ Specific layer quadratic tests __________________
    def test07_quadratic_convergence_gradient_W(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(X.shape[0], X.shape[0])
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = layer_function(X, W + eps * d, b).T.reshape(-1, 1)
            objective2 = layer_function(X, W, b).T.reshape(-1, 1)
            objective3 = layer_function.jacMV_W(X, W, b, (eps * d).reshape(-1, 1),
                                                tanh_grad)  # (X, W, b, V, activation_grad)

            losses.append(norm(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), round(loss_convergence,2), range(len(loss_convergence)), loss_convergence,
                           title=f'Quadratic_layer_convergence_gradient_test_W', semilog=False, label1='Quadratic', label2='Linar', two=False , save_option=False)

    def test08_quadratic_convergence_gradient_X(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(*X.shape)
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = layer_function(X + eps * d, W, b).T.reshape(-1, 1)
            objective2 = layer_function(X, W, b).T.reshape(-1, 1)
            V = (eps * d).T.reshape(-1, 1)
            objective3 = layer_function.jacMV_X(X, W, b, V, tanh_grad)  # (X, W, b, V, activation_grad)

            losses.append(norm(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), round(loss_convergence,2), range(len(loss_convergence)), loss_convergence,
                           title=f'Quadratic_layer_convergence_gradient_test_X', semilog=False, label1='Quadratic', label2='Linar', two=False , save_option=False)

    def test09_quadratic_convergence_gradient_b(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        d = randn(*b.shape)
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            objective1 = layer_function(X, W, b + eps * d).T.reshape(-1, 1)
            objective2 = layer_function(X, W, b).T.reshape(-1, 1)
            objective3 = layer_function.jacMV_b(X, W, b, eps * d, tanh_grad)  # (X, W, b, V, activation_grad)

            losses.append(norm(objective1 - objective2 - objective3))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), round(loss_convergence,2), range(len(loss_convergence)), loss_convergence,
                           title=f'Quadratic_layer_convergence_gradient_test_b', semilog=False, label1='Quadratic', label2='Linar', two=False , save_option=False)

    def test10_transpose_jacMVW(self):
        _, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        u = randn(X.shape[0] * X.shape[1]).reshape(-1, 1)
        v = randn(X.shape[0], X.shape[0]).reshape(-1, 1)

        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        jac_W_v = layer_function.jacMV_W(X, W, b, v, tanh_grad)  # (X, W, b, V, activation_grad)
        jac_T_W_u = layer_function.jacTMV_W(X, W, b, u, tanh_grad)  # (X, W, b, V, activation_grad)
        res = abs(u.T @ jac_W_v - v.T @ jac_T_W_u)
        self.assertTrue(res < 1e-5, msg=f'res = {res}')


    def test11_transpose_jacMVb(self):
        _, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        v = randn(*b.shape)
        u = randn(W.shape[1] * X.shape[1]).reshape(-1, 1)

        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        jac_b_v = layer_function.jacMV_b(X, W, b, v, tanh_grad)  # (X, W, b, V, activation_grad)
        jac_T_b_u = layer_function.jacTMV_b(X, W, b, u, tanh_grad)  # (X, W, b, V, activation_grad)
        res = abs(u.T @ jac_b_v - v.T @ jac_T_b_u)
        self.assertTrue(res < 1e-5, msg=f'res = {res}')

    def test12_transpose_jacMVX(self):
        _, _, X, _, _ = create_C_W_X_d()
        W = randn(X.shape[0], X.shape[0])
        b = randn(X.shape[0]).reshape(-1, 1)
        v = randn(X.shape[0] * X.shape[1]).reshape(-1, 1)
        u = randn(X.shape[0] * X.shape[1]).reshape(-1, 1)

        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        jac_X_v = layer_function.jacMV_X(X, W, b, v, tanh_grad)  # (X, W, b, V, activation_grad)
        jac_T_X_u = layer_function.jacTMV_X(X, W, b, u, tanh_grad)  # (X, W, b, V, activation_grad)
        res = abs(u.T @ jac_X_v - v.T @ jac_T_X_u)
        self.assertTrue(res < 1e-5, msg=f'res = {res}')

    # ________________ full network tests __________________
    def test13_full_network_linear_gradient_test_W(self):
        max_iter = 17
        eps0 = 0.5
        losses = []
        loss_convergence = []
        C_train, C_val, X_train, X_val = data_factory('Swiss')  # options: 'Swiss','PeaksData','GMMData'
        n = X_train.shape[0]
        l = C_train.shape[0]

        layer_function, activation_grad = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W,
                                                    jacTMV_b), tanh_grad
        L = 50
        width = 5
        first_layer_dim = (n, width)
        layer_dim = (width, width)
        output_dim = (width, l)
        model = NeuralNetwork(num_of_layers=L, f=layer_function,
                              activation_grad=activation_grad,  # ReLU_grad,tanh_grad
                              first_layer_dim=first_layer_dim,
                              layer_dim=layer_dim,
                              output_dim=output_dim)

        params = model.params_to_vector(model.params).copy()
        d = randn(*params.shape)
        output1 = objective_soft_max(X=None, W=None, C=C_val, WT_X=model(X_val))

        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            model.set_params(params + eps * d)
            output2 = objective_soft_max(X=None, W=None, C=C_val, WT_X=model(X_val))
            losses.append(abs(output1 - output2))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.4 <= avg_val <= 0.6, msg=f'ans = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), round(loss_convergence,2), range(len(loss_convergence)), loss_convergence,
                           title=f'full_network_linear_gradient_test_W', semilog=False, label1='Linar', label2='Quadratic', two=False , save_option=False)

    def test14_full_network_linear_gradient_test_W(self):
        max_iter = 15
        eps0 = 1
        losses = []
        loss_convergence = []
        C_train, C_val, X_train, X_val = data_factory('GMMData')  # options: 'Swiss','PeaksData','GMMData'
        n = X_train.shape[0]
        l = C_train.shape[0]


        layer_function, activation_grad = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W,
                                                    jacTMV_b), tanh_grad
        L = 2
        width = 5
        first_layer_dim = (n, width)
        layer_dim = (width, width)
        output_dim = (width, l)
        model = NeuralNetwork(num_of_layers=L, f=layer_function,
                              activation_grad=activation_grad,  # ReLU_grad,tanh_grad
                              first_layer_dim=first_layer_dim,
                              layer_dim=layer_dim,
                              output_dim=output_dim)

        params = model.params_to_vector(model.params).copy()

        X_val = np.array([[0.57060438,0.61364776,0.58852160,0.68732345,0.58038837,0.58508348],
                          [0.81162298,0.69123715,0.74537230,0.65157044,0.79017442,0.75268739],
                          [0.68581861,0.85270423,0.78290349,0.96603888,0.72208542,0.78569037],
                          [0.019703234,0.045418661,0.023746030,0.024757080,-0.0090582278,0.024008246],
                          [0.17623653,0.36377695,0.17722517,0.15017267,0.057988834,0.27432564]])

        C_val = C_val[:, 0:6]

        params2 = np.array([-0.1,-0.6,-0.1,-0.6,-0.1,0.2,0.7,0.2,0.7,0.2,-0.3,-0.8,-0.3,-0.8,-0.3,0.4,0.9,0.4,0.9,0.4,0.5,1,0.5,1,0.5,0.1,0.2,0.3,0.4,0.5,
                            0.1,0.6,0.1,0.6,0.1,0.2,0.7,0.2,0.7,0.2,0.3,0.8,0.3,0.8,0.3,0.4,0.9,0.4,0.9,0.4,0.5,1,0.5,1,0.5]).reshape(55,1)
        params = params2
        model.set_params(params)

        #d = randn(params.shape[0] * params.shape[1], 1)
        #d = d / norm(d)

        d = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
                      0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1]).reshape(55,1)

        output2 = model(X_val)
        objective2 = objective_soft_max(X=None, W=None, C=C_val, WT_X=output2)
        d_dw = model.backward(objective_soft_max_gradient_X(X=None, W=model.params[L - 1][0], C=C_val, WT_X=output2),
                              objective_soft_max_gradient_W(X=model.layers_inputs[L - 1], W=None, C=C_val,
                                                            WT_X=output2))
        # d_dw = d_dw / norm(d_dw)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            new_params = params + eps * d
            model.set_params(new_params)
            output1 = model(X_val)
            objective1 = objective_soft_max(X=None, W=None, C=C_val, WT_X=output1)
            objective3 = eps * (d.T @ d_dw)

            losses.append(abs(objective1 - objective2 - objective3.item(0)))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        print (loss_convergence)
        avg_val = average(round(loss_convergence[-1:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')

        show_and_save_plot(range(len(loss_convergence)), loss_convergence, range(len(loss_convergence)), loss_convergence,
                           title=f'full_network_Quadratic_gradient_test_W', semilog=False, label1='Quadratic', label2='Linar', two=False , save_option=False)





    def test15_full_network_linear_gradient_test_W(self):
        max_iter = 10
        eps0 = 0.125
        losses = []
        loss_convergence = []
        C_train, C_val, X_train, X_val = data_factory('PeaksData')  # options: 'Swiss','PeaksData','GMMData'

        # X_val = np.array([[0.57647061, 0.67058825, -0.15294118, 0.69411767, 0.38823530, -0.2, 0.011764706,
        #                    0.29411766, 0.76470590, 0.76470590],
        #                   [-1.8941176, -1.3058823, -1.9411764, -1.7058823, -2.0823529, -1.7529411, -1.3529412,
        #                    -1.6823529, -1.7058823, -1.4000000]])
        #
        # C_val = C_val[:, 0:10]

        n = X_train.shape[0]
        l = C_train.shape[0]
        layer_function = Function(tanh_F, jacMV_X, jacMV_W, jacMV_b, jacTMV_X, jacTMV_W, jacTMV_b)
        L = 2
        model = NeuralNetwork(num_of_layers=L, f=layer_function, activation_grad=tanh_grad, layer_dim=(n, n),
                              output_dim=(n, l))
        params = model.params_to_vector(model.params).copy()
        #params_ilan = np.array([0.2, -0.3, 0.5, 1.7, 0.8, 0.4, 0.2, -0.3, 0.5, 1.7, 0, 2.1, 0.4, -0.4, 0.8, -0.9]).reshape(16,1)
        #params = params_ilan
        model.set_params(params)
        d = randn(params.shape[0] * params.shape[1], 1)

        idxx = 0
        sum = 0
        nn ,mm =model.layer_dim

        for i in range (L-1):
            idxx_end1 = idxx + nn*mm
            idxx_end2 = idxx_end1 + mm
            sum = sum + np.power(norm(d[idxx:idxx_end1].T), 2) + np.power(norm(d[idxx_end1:idxx_end2].T), 2)
            idxx = idxx_end2
        z = np.power(norm(d[idxx:].T), 2)
        sum = sum + np.power(norm(d[idxx:].T), 2)

        d = d / np.power(sum, 0.5)

        #d = (np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).T).reshape(16,1)
        output2 = model(X_val)
        objective2 = objective_soft_max(X=None, W=None, C=C_val, WT_X=output2)
        d_dw = model.backward(
            objective_soft_max_gradient_X(X=None, W=model.params[L - 1][0], C=C_val, WT_X=output2),
            objective_soft_max_gradient_W(X=model.layers_inputs[L - 1], W=None, C=C_val,
                                          WT_X=output2))
        # d_dw = d_dw / norm(d_dw)
        for i in range(max_iter):
            eps = eps0 * (0.5 ** i)
            new_params = params + eps * d
            model.set_params(new_params)
            output1 = model(X_val)
            objective1 = objective_soft_max(X=None, W=None, C=C_val, WT_X=output1)
            objective3 = eps * (d.T @ d_dw)

            losses.append(abs(objective1 - objective2 - objective3.item(0)))

        for j in range(1, len(losses)):
            loss_convergence.append(losses[j] / losses[j - 1])

        avg_val = average(round(loss_convergence[-5:], 4))
        self.assertTrue(0.2 <= avg_val <= 0.3, msg=f'ans = {avg_val}')
        # todo add plot

        if __name__ == '__main__':
            unittest.main()