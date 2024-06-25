
import numpy as np

class ReluNB:

    def __init__(self):
        
        # z comes from linear layer
        # a = relu(z)
        self.dLdZ = None

        self.dadz = None

    def init_grad(self, x):
        
        ones_iff_pos = np.where(x > 0, 1, 0)

        self.dadz = np.diag(ones_iff_pos)
        #print(self.dadz)

    def forward(self, x):

        self.init_grad(x)

        return np.maximum(x, 0)
    
    def backward(self, dLda):

        self.dLdZ = np.tensordot(dLda, self.dadz, axes=([0], [0]))