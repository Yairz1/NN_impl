class SGD:
    def __init__(self, batch_size, epochs):
        self.batch_size = batch_size
        self.epochs = epochs
        self.convergence_history = list()

    def optimize(self, F, X, W):
        pass

    def get_history(self):
        return self.convergence_history
