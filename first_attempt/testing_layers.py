import numpy as np
from linear_layer import *
from relu_layer import *
from sigmoid_layer import *
from bce_loss import *

feat_dim = 5

x = np.random.randn(feat_dim)
#print(x)

l1 = LinearNB(feat_dim, out_dim = 10)
l2 = LinearNB(in_dim=10, out_dim=1)
r1 = ReluNB()
s1 = SigmoidNB()
b1 = BCELossNB()

print(x)
y = l1.forward(x)
print(y)
y2 = r1.forward(y)
print(y2)
y3 = l2.forward(y2)
print(y3)
y4 = s1.forward(y3)
print(y4)

loss = b1.forward(y4, 1.0)
print(loss)

print("Backward Pass")

# now, caculate the parameter derivatives wrt loss
dLdy4 = b1.backward()
print(dLdy4)

s1.backward(dLdy4)
dLdy3 = s1.dLdZ

print(dLdy3)
l2.backward(dLdy3)

dLdy2 = l2.dl



#l1.backward(dLdZ=dLdy)