
from tensor import Tensor
from layer import Layer
import numpy as np

class Linear(Layer):

    def __init__(self, in_dim, out_dim):

        # TODO: add kaiming initialization

        self.W = Tensor(np.random.randn(out_dim, in_dim))
        self.b = Tensor(np.zeros((1, out_dim)))

        self.params = {
            "W": self.W,
            "b": self.b
        }

    def forward(self, X: Tensor):
        
        # x in (batch, in_dim)
        batch_size = X.value.shape[0]
        
        # (batch, in_dim) * (in_dim, out_dim) = (batch, out_dim)
        # h in (batch, out_dim)
        h = X.dot(self.W.transpose())

        # b only in out_dim, so we repeat the rows
        b_broadcasted = self.b.repeat_rows(batch_size)
        h = h.add(b_broadcasted)

        return h
    
    def get_params(self):
        return self.params