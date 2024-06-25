import numpy as np

# NB = no batching

class LinearNB:

    def __init__(self, in_dim, out_dim):

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W = np.random.randn(out_dim, in_dim)
        self.b = np.random.randn(out_dim)

        # those are important for the optimization
        self.dLdW = None
        self.dLdb = None

        # this layer produces a z in R^(out_dim)
        # and this can influence the L (a scalar number)
        self.dzdW = None
        self.dzdb = None

    def init_grad(self, x):
        
        # if we have a single scalar, every matrix entry can influence it
        # the derivative would be a matrix

        # but we have out_dim scalars so we have a third order tensor
        self.dzdW = np.zeros((self.out_dim, self.out_dim, self.in_dim))

        # every one of the out_dim outputs gets one matrix of how much the matrix entries influence it
        for i in range(self.out_dim):

            # the ith output is only influenced by the ith row of the matrix
            self.dzdW[i, i] = x

        self.dzdb = np.eye(self.out_dim, self.out_dim)

    def forward(self, x):
        
        # collect information for the backward pass
        self.init_grad(x)
        
        return np.dot(self.W, x) + self.b

    def backward(self, dLdZ):
        
        self.dLdW = np.tensordot(dLdZ, self.dzdW, axes=([0], [0]))
        self.dLdb = np.tensordot(dLdZ, self.dzdb, axes=([0], [0]))