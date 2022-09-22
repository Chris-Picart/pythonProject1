import numpy as np


x3 = .5
x2 = .5 # initial guess values
a = 1 # intital value for alpha
t = 0.1 # learning rate
iterations = 1000
x = np.array([x2, x3])  # initial guess

def f(x):
    f = np.array((-2 + 2 * x[0] + 3 * x[1]) ** 2 + (x[0]) ** 2 + (1 - x[1]) ** 2)
    return f


def g(x):
    g = np.array([(10 * x[0] + 12 * x[1] - 8), (12 * x[0] + 20 * x[1] - 14)])
    return g


def Xn(x):
    xn = x - (a * g(x))
    return xn

def phi(a):
    phi = f(x) - (t * g_t * g(x) * a)
    return phi
g_t = g(x).transpose()
print(phi(x))
def gdinexact(x2,x3):
    a = 1
    for i in range(iterations):
        if f(Xn(x)) > phi(a):
            a = a*.5
    return(x2,x3)

print(gdinexact(x2,x3))
#print("f {}, iteration {}, x {}".format(f(x), i, x))

