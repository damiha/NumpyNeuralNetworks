
from neural_network import *
from linear_layer import *
from relu_layer import *
from sigmoid_layer import *
from bce_loss import *

nn = NeuralNetworkNB(
    layers=[
        LinearNB(2, 4),
        ReluNB(),
        LinearNB(4, 1),
        SigmoidNB()
    ],
    loss_function=BCELossNB()
)

# XOR problem
x1 = np.array([0.0, 0])
x2 = np.array([0, 1.0])
x3 = np.array([1.0, 0])
x4 = np.array([1.0, 1])

y1 = nn.forward(x1, 0)
print(y1)

nn.backward()

