import numpy as np
from linear_layer import *
from relu_layer import *
from sigmoid_layer import *

feat_dim = 5

x = np.random.randn(feat_dim)
#print(x)

l1 = LinearNB(feat_dim, out_dim = 10)
l2 = LinearNB(in_dim=10, out_dim=1)
r1 = ReluNB()
s1 = SigmoidNB()

y = l1.forward(x)
y2 = r1.forward(y)
y3 = l2.forward(y2)
y4 = s1.forward(y3)

loss = np.sum(y2)

# now, caculate the parameter derivatives wrt loss
dLdy2 = np.ones_like(y2)
print(dLdy2.shape)

r1.backward(dLdy2)
dLdy = r1.dLdZ
print(dLdy.shape)

l1.backward(dLdy)
print(l1.dLdW.shape)
print(l1.dLdb.shape)


#l1.backward(dLdZ=dLdy)