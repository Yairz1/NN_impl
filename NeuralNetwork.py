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
                                         output_dim)  # a dict of the form [#layer:(W,b,bias:bool)]
        self.layers_inputs = dict()
        self.f = f
        self.activation_grad = activation_grad
        self.gradient = dict()  # a dict of the form [#layer:grad_w,grad_b]

    def forward(self, x):
        for layer, (W, b) in self.params.items():
            self.save(layer, x)
            x = self.f(x, W, b)
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

        return self.gradient

    def params_to_vector(self):
        vectorized_grad = array([]).reshape(-1, 1)
        for layer, (W, b, isBiased) in self.params.items():
            if isBiased:
                pass
            # todo: complete
            vectorized_grad = vstack([vectorized_grad, W.reshape(-1, 1)])
        return vectorized_grad.reshape(-1)

    def save(self, layer, x):
        self.layers_inputs[layer] = x

    def gradient_reset(self):
        pass  # self.gradient = zeros()

    def __call__(self, x):
        return self.forward(x)
