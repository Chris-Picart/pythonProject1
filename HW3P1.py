import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
# gradient descent
c = 0
A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True) #assigning initial guess of A here, A12 = 1, A21 = 1
def gdinexact(A):
    iterations = 100 # number of iterations
    t = .5
    a = 1
    def s(A): # this function defines step, we are minimizing hence, - sign
        gd = -g(A) # step direction for gradient descent
        #newt = -np.matmul(np.linalg.inv(H),g(x)).flatten() # step direction for newtons method
        return gd # return newt to use newtons method and gd to use gradient descent

    def d(A): # this is function being minimized
        x1 = torch.tensor(np.array([[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]), requires_grad=False, dtype=torch.float32)
        x2 = np.flip(x1, axis=1).copy() # defines x2 as 1-x1
        T = 20
        p_water = 10 ** (a[0, 0] - a[0, 1] / (T + a[0, 2]))  # equation for p1 sat based on water system
        p_dio = 10 ** (a[1, 0] - a[1, 1] / (T + a[1, 2]))  # equation for p2 sat based on 1,4 dioxane system
        P = np.array([[28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]])
        P = torch.tensor(P, requires_grad=False, dtype=torch.float32)
        d = x1*torch.exp(A[0] * (A[1] * x2 / (A[0] * x1 + A[1] * x2)) ** 2) * p_water + \
        x2 * torch.exp(A[1] * (A[0] * x1 / (A[0] * x1 + A[1] * x2)) ** 2) * p_dio
        return d

    def g(A): # gradient of the function being minimized
        g = A.grad
        return g

    def phi(a): # phi function being defined to compare function value at x-
        gx = g(A)
        gt = gx.transpose()
        st = s(A).transpose()
        phi = d(A) + (t * np.matmul(gt,st) * a) # step function
        return phi

    def linesearch(A,a):
        while d(A+a*s(A)) > phi(a): # -g(x) is step direction
            a = 0.5*a
        return a


    #for i in range(iterations):
    while abs(g(A)) > 1e-3:
        a = linesearch(A,a)
        A -= a*s(A)  #decrease A by multiplying current value and adding (alpha*step)


       # A.grad.zero_()

    return A
print("final solution A=",(gdinexact(A)))
