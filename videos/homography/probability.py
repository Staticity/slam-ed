import numpy as np

class Gaussian:
    def __init__(self, mean=0.0, sigma=1):
        self.u = mean
        self.o = sigma

    def sample(self):
        return np.random.normal(self.u, self.o)

