import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from IPython import display

X1 = np.array([[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]])
X2 = np.flip(X1, axis=1).copy() # defines x2 as 1-x1
a = np.array(([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]]))
T = 20

p_water = 10 ** (a[0,0]-a[0,1] / (T + a[0,2])) # equation for p1 sat based on water system
p_dio = 10 ** (a[1,0] - a[1,1] / (T+ a[1,2])) # equation for p2 sat based on 1,4 dioxane system
# ^ given in log(psat) so need to be reverted by placing in expoential for 10
P = np.array ([[28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]])
P = torch.tensor(P, requires_grad=False, dtype=torch.float32)

A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True) #assigning initial guess of A here, A12 = 1, A21 = 1
#creating the variable A as a tensor under the conidtion that is requires a gradient

x1 = torch.tensor(X1, requires_grad=False, dtype=torch.float32) # importing data points as Tensors (no gradient needed)
x2 = torch.tensor(X2, requires_grad=False, dtype=torch.float32)

a = .0001 # step size for gradient descent

P_pred = x1 * torch.exp(A[0] * (A[1] * x2 / (A[0] * x1 + A[1] * x2)) ** 2) * p_water + \
         x2 * torch.exp(A[1] * (A[0] * x1 / (A[0] * x1 + A[1] * x2)) ** 2) * p_dio  # equation used to predict pressure value

loss = (P_pred - P) ** 2  # element-wise subtraction of two vectors
loss = loss.sum()  # loss is defined as the square error so now sum it

loss.backward()  # calling the line computes gradient for current guess
C = torch.norm(A.grad).item()

while C >= 1e-3: # implement gradient descent condition to reduce gradient value to acceptable limit

    P_pred = x1*torch.exp(A[0] * (A[1] * x2 / (A[0] * x1 + A[1] * x2)) ** 2) * p_water + \
        x2 * torch.exp(A[1] * (A[0] * x1 / (A[0] * x1 + A[1] * x2)) ** 2) * p_dio #equation used to predict pressure value

    loss = (P_pred - P) ** 2 # element-wise subtraction of two vectors
    loss = loss.sum() # loss is defined as the square error so now sum it

    loss.backward()  # calling the line computes gradient for current guess
    C = torch.norm(A.grad).item()
    with torch.no_grad():
        A -= a * A.grad #A.grad will give current gradient value
        # ^ this line means A = A - a*A.grad, gives new estimation of A using gradient descent
        A.grad.zero_() # now zero the gradient or it will be a summation


print('Estimation of A12 and A21 is;', A)
print('final loss is:', loss.data.numpy())

P_pred = P_pred.detach().numpy()[0]
P = P.detach().numpy()[0]
x1 = x1.detach().numpy()[0]

plt.plot(x1, P_pred, label='Predicted Pressure')
plt.plot(x1, P, label='Actual Pressure')
plt.xlabel('x1')
plt.ylabel('Pressure')
plt.legend()
plt.title('Comparison between predicted pressure and actual pressure')
plt.show()

