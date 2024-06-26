
class Optimizer:

    def __init__(self, parameters):

        # every optimizer needs to know which parameters to optimize
        self.parameters = parameters

    # this differs for every optimizer
    def step(self):
        raise NotImplementedError("'step' missing")
    
    def zero_grad(self):
        
        for pname, p in self.parameters.items():
            p.grad = 0