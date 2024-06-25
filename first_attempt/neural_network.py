
from linear_layer import *

class NeuralNetworkNB:

    def __init__(self, layers, loss_function):

        self.layers = layers
        self.loss_function = loss_function

        self.mode = "train"
        assert(self.mode in ["train", "test"])

    def forward(self, x, y_true=None):

        assert (self.mode == "test" or y_true is not None)
        
        curr_input = x
        curr_output = None

        for layer in self.layers:
            curr_output = layer.forward(curr_input)
            curr_input = curr_output

        y_pred = curr_output
        
        if self.mode == "test":
            return y_pred
        
        else:
            loss_value = self.loss_function.forward(y_pred, y_true)
            return y_pred, loss_value
    
    def backward(self):
        
        dLoss = self.loss_function.backward()

        for layer in reversed(self.layers):

            print(dLoss)
            print(type(layer))

            layer.backward(dLoss)

            if type(layer) == LinearNB:

                # don't propagate the gradients of the bias back because bias can't be influenced by layers coming before it
                dLoss = layer.dLdW

            else:
                dLoss = layer.dLdZ





        