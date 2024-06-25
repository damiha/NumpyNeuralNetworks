
import numpy as np

class SigmoidNB:

    def __init__(self):

        self.dadz = None

        self.dLdZ = None

    def sigmoid(self, x):

        e_to_x = np.exp(x)
        return e_to_x / (1.0 + e_to_x)

    def init_grad(self, x):

        sigmoid_x = self.sigmoid(x)

        self.dadz = np.diag(sigmoid_x * (1.0 - sigmoid_x))

    def forward(self, x):

        self.init_grad(x)

        return self.sigmoid(x)

    # dLda = how value after that influences the loss
    def backward(self, dLda):

        # goal is to set dLdZ (how does the unactivated z influence the loss)
        self.dLdZ = np.tensordot(dLda, self.dadz, axes=([0], [0]))