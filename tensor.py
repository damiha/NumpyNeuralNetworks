import numpy as np

class Tensor:

    def __init__(self, value: np.ndarray):

        self.value = value
        self.requires_grad = False

        self.grad = np.zeros_like(value)

        self.prev = set()

        self._backward = lambda: None

    def dot(self, other):

        out = Tensor(self.value.dot(other.value))
        out.prev = {self, other}

        def _backward():

            self.grad += (out.grad @ other.value.T)
            other.grad += (self.value.T @ out.grad)

        out._backward = _backward

        out.requires_grad = (self.requires_grad or other.requires_grad)

        return out
    
    def add(self, other):

        out = Tensor(self.value + other.value)
        out.prev = {self, other}

        def _backward():

            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        out.requires_grad = (self.requires_grad or other.requires_grad)

        return out
    
    def repeat_cols(self, k):

        out = Tensor(np.repeat(self.value, repeats=(k, ), axis=1))
        out.prev = {self}

        def _backward():

            self.grad += out.grad @ np.ones((k, 1))

        out._backward = _backward
        out.requires_grad = self.requires_grad

        return out
    
    def repeat_rows(self, k):

        out = Tensor(np.repeat(self.value, repeats=(k, ), axis=0))
        out.prev = {self}

        def _backward():

            self.grad += np.ones((1, k)) @ out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad

        return out
    
    def transpose(self):

        out = Tensor(self.value.T)
        out.prev = {self}

        def _backward():

            self.grad += out.grad.T

        out._backward = _backward
        out.requires_grad = self.requires_grad

        return out
    
    def mul(self, c: float):

        out = Tensor(c * self.value)
        out.prev = {self}

        def _backward():

            self.grad += (c * out.grad)

        out._backward = _backward
        out.requires_grad = self.requires_grad

        return out
    
    # reduces everything to a scalar
    def sum(self):

        out = Tensor(self.value.sum())
        out.prev = {self}

        def _backward():

            self.grad += out.grad * np.ones_like(self.value)

        out._backward = _backward
        out.requires_grad = self.requires_grad

        return out
    
    # not the best from a SWE standpoint
    def relu(self):

        out = Tensor(np.maximum(self.value))
        out.prev = {self}

        def _backward():

            self.grad += (out.grad * (self.value > 0))

        out._backward = _backward
        out.requires_grad = self.requires_grad

        return out
    
    def backward(self):

        assert(type(self.value) == np.float64)

        # build the topological sort first
        topo = []
        visited = set()

        def build_topo(v):
            if not v in visited:

                visited.add(v)

                for c in v.prev:
                    build_topo(c)

                topo.append(v)

        build_topo(self)

        self.grad = 1

        # now, the first node that should be called _backward on (self) is in the back of the list
        for v in reversed(topo):
            v._backward()
                
            


    

    


