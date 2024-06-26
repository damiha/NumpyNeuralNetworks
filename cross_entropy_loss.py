
from tensor import Tensor
import numpy as np

class CELossWithLogits:

    def __init__(self):

        pass

    def softmax(self, logits: np.ndarray):
        # logits in (batch, n_classes)
        unnormalized_probs = np.exp(logits)
        return unnormalized_probs / np.sum(unnormalized_probs, axis=1, keepdims=True)

    def forward(self, logits: Tensor, y_true: Tensor):

        batch_size = y_true.value.shape[0]

        # y_true in (batch_dim, n_classes)

        y_pred = self.softmax(logits.value)

        loss_val = -np.sum(np.log(y_pred) * (y_true.value)) / batch_size

        loss = Tensor(loss_val)
        loss.prev = {logits}

        def _backward():
            # https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
            logits.grad += (y_pred - y_true.value) / batch_size

        loss._backward = _backward
        loss.requires_grad = logits.requires_grad

        return loss

