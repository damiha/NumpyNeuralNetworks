
from tensor import Tensor

class MSELoss:

    def forward(self, y_pred: Tensor, y_true: Tensor):
        
        # N is the batch size
        N = y_pred.value.shape[0]
 
        return y_pred.sub(y_true).pow(2).sum().mul(1.0 / N)

        