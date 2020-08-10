import function
from numpy.random import randn
from numpy import zeros, identity, vstack, array, ones


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
            x[-1:, :] = ones(x.shape[1])  # for bias
            # todo last iteration shouldnt use activation function because of the gradients
        return x

    def backward(self, loss_gradient_x, loss_gradient_w):  # dx,dw
        assert bool(self.layers_inputs), "You must run forward before backward"
        self.gradient[self.num_layers - 1] = loss_gradient_w
        v = loss_gradient_x
        for layer in reversed(range(self.num_layers - 1)):
            self.gradient[layer] = self.f.f_grad_W_mul_V(self.layers_inputs[layer], self.params[layer], v, self.sigma)
            v = self.f.f_grad_X_mul_V(self.layers_inputs[layer], self.params[layer], v, self.sigma)

        return self.gradient  # self.params_to_vector()

    def params_to_vector(self):
        vectorized_grad = array([]).reshape(-1, 1)
        for layer, w in self.params.items():
            vectorized_grad = vstack([vectorized_grad, w.reshape(-1, 1)])
        return vectorized_grad.reshape(-1)

    def save(self, layer, x):
        self.layers_inputs[layer] = x

    def gradient_reset(self):
        pass  # self.gradient = zeros()

    def __call__(self, x):
        return self.forward(x)
