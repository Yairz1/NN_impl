import function
from numpy.random import randn
from numpy import zeros, identity, vstack, array, ones


def init_random_params(num_layers, layer_dim, output_param_dim):
    '''

    :param num_layers:
    :param layer_dim:
    :param output_param_dim:
    :return: a dict of the form [#layer:(W,b,bias:bool)]
    '''
    params = dict()
    n, m = layer_dim
    for layer in range(num_layers - 1):
        params[layer] = randn(n, m), randn(m, 1)
    params[layer + 1] = randn(*output_param_dim), zeros((output_param_dim[1], 1))
    return params


class NeuralNetwork:
    def __init__(self, num_of_layers, f, activation_grad, layer_dim, output_dim):
        self.num_layers = num_of_layers
        self.output_dim = output_dim
        self.params = init_random_params(num_of_layers, layer_dim,
                                         output_dim)  # a dict of the form [#layer:(W,b)]
        self.layers_inputs = dict()
        self.f = f
        self.activation_grad = activation_grad
        self.gradient = dict()  # a dict of the form [#layer:grad_w,grad_b]
        self.layer_dim = layer_dim
        self.output_dim = output_dim

    def forward(self, x):
        for layer, (W, b) in self.params.items():
            self.save(layer, x)
            if layer == self.num_layers - 1:
                x = self.f(x, W, b, False)
            else:
                x = self.f(x, W, b, True)
        return x

    def backward(self, loss_gradient_x, loss_gradient_w):  # dx,dw
        assert bool(self.layers_inputs), "You must run forward before backward"
        self.gradient[self.num_layers - 1] = loss_gradient_w.T.reshape(-1, 1), 0
        v = loss_gradient_x.T.reshape(-1, 1)
        for layer in reversed(range(self.num_layers - 1)):
            W, b = self.params[layer]
            self.gradient[layer] = (self.f.jacTMV_W(self.layers_inputs[layer],
                                                    W,
                                                    b,
                                                    v,
                                                    self.activation_grad),
                                    self.f.jacTMV_b(self.layers_inputs[layer],
                                                    W,
                                                    b,
                                                    v,
                                                    self.activation_grad))

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
        last_w, _ = params[layer + 1]
        return vstack([vectorized_grad, last_w.T.reshape(-1, 1)])

    def vector_to_params(self, W):
        params = dict()
        n, m = self.layer_dim
        W_i_start_idx = 0
        for layer in range(self.num_layers - 1):  # layer:(W,b)
            W_i_end_idx = W_i_start_idx + n * m
            b_i_start_idx = W_i_end_idx
            b_i_end_idx = b_i_start_idx + m  # b length = m
            params[layer] = (W[W_i_start_idx:W_i_end_idx].reshape(m, n).T,
                             W[b_i_start_idx:b_i_end_idx].reshape(-1, 1))
            W_i_start_idx = b_i_end_idx
        params[layer + 1] = W[-m * n:].reshape(m, n).T, zeros((self.output_dim[1], 1))

        return params

    def set_params(self, W):
        self.params = self.vector_to_params(W)

    def save(self, layer, x):
        self.layers_inputs[layer] = x

    def gradient_reset(self):
        pass  # self.gradient = zeros()

    def __call__(self, x):
        return self.forward(x)
