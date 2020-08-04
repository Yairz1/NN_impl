import function
from numpy.random import randn
from numpy import zeros, identity


def init_random_params(num_layers, layer_dim, output_param_dim):
    params = dict()
    for layer in range(num_layers - 1):
        params[layer] = randn(*layer_dim)
    params[layer + 1] = randn(*output_param_dim)
    return params


class NeuralNetwork:
    def __init__(self, num_of_layers, f, sigma, layer_dim, output_dim):
        self.num_layers = num_of_layers
        self.output_dim = output_dim
        self.params = init_random_params(num_of_layers, layer_dim, output_dim)
        self.layers_inputs = dict()
        self.f = f
        self.sigma = sigma
        self.gradient = dict()

    def forward(self, x):
        for layer, W in self.params.items():
            self.save(layer, x)
            x = self.f(x, W)
        return x


    def backward(self, x):  # dx,dw
        # self.gradient[self.num_layers - 1] = identity(*self.output_dim)
        # self.gradient[self.num_layers - 2] = grad_w(self.layers_inputs[...],self.params[...])
        # v = grad_x(self.layers_inputs[...],self.params[...])
        # for layer, W in reversed(range(self.num_layers - 2)):
        #     self.gradient[layer] =  grad_w(self.layers_inputs[...],self.params[...]) * v
        #     v = grad_x(self.layers_inputs[layer-1], self.params[layer-1]) * v

        return x

    def params_to_vector(self):
        pass  # todo reshape

    def save(self, layer, x):
        self.layers_inputs[layer] = x

    def gradient_reset(self):
        self.gradient = zeros()

    def __call__(self, x):
        return self.forward(x)
