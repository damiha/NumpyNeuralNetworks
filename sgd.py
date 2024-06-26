
from optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, parameters, lr):

        super().__init__(parameters)
        self.lr = lr

    def step(self):

        for pname, p in self.parameters.items():
            
            # sgd = stochastic gradient descent, so we walk in the direction of the negative gradient
            p.value = p.value - (self.lr * p.grad)