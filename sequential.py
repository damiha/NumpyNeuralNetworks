
from tensor import Tensor
from layer import Layer

class Sequential(Layer):

    def __init__(self, *layers):

        self.layers = layers

        self.params = self.collects_params(layers)

    def collects_params(self, layers):
        
        def rename_params(l, i):
            return {f"{str(l)}{i}{pname}": p for pname, p in l.get_params().items()}
        
        all_params = {}

        for i, l in enumerate(layers):
            all_params = dict(all_params, **rename_params(l, i))

        return all_params

    def get_params(self):
        return self.params

    def forward(self, x: Tensor):

        for layer in self.layers:

            x = layer.forward(x)

        return x