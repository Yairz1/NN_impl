import activation
from numpy.random import randn


def init_random_params(num_layers, layer_dim):
    params = dict()
    for layer in range(num_layers):
        params[layer] = randn(*layer_dim)
    return params


class NeuralNetwork:
    def __init__(self, num_of_layers, f, sigma, layer_dim, output_dim):
        self.num_layers = num_of_layers
        self.params = init_random_params(num_of_layers, layer_dim)
        self.f = f
        self.sigma = sigma

    def forward(self, x):
        for layer, W in self.params.items():
            x = self.sigma(self.f(x, W))
        return x

    def backward(self, x):
        return x

    def params_to_vector(self):
        pass  # todo reshape
