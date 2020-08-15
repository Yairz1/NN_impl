import function
from numpy.random import randn
from numpy import zeros, identity, vstack, array, ones
from numpy.linalg import norm


def init_random_params(num_layers, first_layer_dim, layer_dim, output_param_dim):
    '''

    :param num_layers:
    :param layer_dim:
    :param output_param_dim:
    :return: a dict of the form [#layer:(W,b,bias:bool)]
    '''
    params = dict()
    params[0] = randn(*first_layer_dim), randn(first_layer_dim[1], 1)
    n, m = layer_dim
    for layer in range(1, num_layers - 1):
        params[layer] = randn(n, m), randn(m, 1)
    params[num_layers-1] = randn(*output_param_dim), zeros((output_param_dim[1], 1))
    return params


class NeuralNetwork:
    def __init__(self, num_of_layers, f, activation_grad, first_layer_dim, layer_dim, output_dim):
        self.num_layers = num_of_layers
        self.output_dim = output_dim
        self.params = init_random_params(num_of_layers, first_layer_dim, layer_dim,
                                         output_dim)  # a dict of the form [#layer:(W,b)]
        self.layers_inputs = dict()
        self.f = f
        self.activation_grad = activation_grad
        self.gradient = dict()  # a dict of the form [#layer:grad_w,grad_b]
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.first_layer_dim = first_layer_dim

    def forward(self, x):
        for layer, (W, b) in self.params.items():
            self.save(layer, x)
            isLastLayer = (layer == self.num_layers - 1)  # no bias for the last layer
            x = self.f(x, W, b, not isLastLayer)

        return x

    def backward(self, loss_gradient_x, loss_gradient_w):  # dx,dw
        assert bool(self.layers_inputs), "You must run forward before backward"
        self.gradient[self.num_layers - 1] = loss_gradient_w.T.reshape(-1, 1), 0
        v = loss_gradient_x.T.reshape(-1, 1)
        for layer in reversed(range(self.num_layers - 1)):
            W, b = self.params[layer]
            grad_W, grad_b = (self.f.jacTMV_W(self.layers_inputs[layer],
                                              W,
                                              b,
                                              v,
                                              self.activation_grad),
                              self.f.jacTMV_b(self.layers_inputs[layer],
                                              W,
                                              b,
                                              v,
                                              self.activation_grad))
            self.gradient[layer] = grad_W / norm(grad_W), grad_b / norm(grad_b)
            v = self.f.jacTMV_X(self.layers_inputs[layer],
                                W,
                                b,
                                v,
                                self.activation_grad)

        return self.params_to_vector(self.gradient)

    def params_to_vector(self, params):
        vectorized_grad = array([]).reshape(-1, 1)
        for layer in range(self.num_layers - 1):  # layer:(W,b)
            W, b = params[layer]
            vectorized_grad = vstack([vectorized_grad, W.T.reshape(-1, 1), b.reshape(-1, 1)])
        last_w, _ = params[self.num_layers - 1]
        return vstack([vectorized_grad, last_w.T.reshape(-1, 1)])

    def vector_to_params(self, W):
        params = dict()
        f_n, f_m = self.first_layer_dim
        params[0] = (W[:f_n * f_m].reshape(f_m, f_n).T,
                     W[f_n * f_m:(f_n * f_m) + f_m].reshape(-1, 1))
        n, m = self.layer_dim
        W_i_start_idx = (f_n * f_m) + f_m
        for layer in range(1, self.num_layers - 1):  # layer:(W,b)
            W_i_end_idx = W_i_start_idx + n * m
            b_i_start_idx = W_i_end_idx
            b_i_end_idx = b_i_start_idx + m  # b length = m
            params[layer] = (W[W_i_start_idx:W_i_end_idx].reshape(m, n).T,
                             W[b_i_start_idx:b_i_end_idx].reshape(-1, 1))
            W_i_start_idx = b_i_end_idx
        params[self.num_layers - 1] = W[W_i_start_idx:].reshape(self.output_dim[1], self.output_dim[0]).T, \
                            zeros((self.output_dim[1], 1))
        return params

    def set_params(self, W):
        self.params = self.vector_to_params(W)

    def save(self, layer, x):
        self.layers_inputs[layer] = x

    def gradient_reset(self):
        pass  # self.gradient = zeros()

    def __call__(self, x):
        return self.forward(x)
