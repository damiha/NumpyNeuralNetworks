from tensor import Tensor
from layer import Layer

class ReLU(Layer):

    def forward(self, x: Tensor):
        return x.relu()
    
    def get_params(self):
        return {}
