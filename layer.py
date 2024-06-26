
from tensor import Tensor

class Layer:

    def forward(self, x: Tensor):
        raise NotImplementedError("'forward' missing")

    def __str__(self):
        return type(self).__name__
    
    def get_params(self):
        raise NotImplementedError("'get_params' missing")