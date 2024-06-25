
import numpy as np

class BCELossNB:

    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_pred, y_true):

        # negative log likelihood of p_i^(y_true) * (1 - p_i)^(1 - y_true)

        # log likelihood = y_true * log(p_i) + (1 - y_true) * log(1 - p_i)

        # negative

        # derivative wrt the y_pred is a vector field (multi label classification problem in the general case)

        # we need to find a jacobi matrix

        # save for the backward pass
        self.y_true = y_true
        self.y_pred = y_pred

        return -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def backward(self):

        
        # what's the derivative for one component?
        # wrt to y_pred

        dLdy = -(self.y_true / self. y_pred + ((1.0 - self.y_true) / (1.0 - self.y_pred)) * (-1))

        # for multi-class classification problems?
        #dLdy = np.diag(dLdy)

        return dLdy