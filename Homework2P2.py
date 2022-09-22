import numpy as np
import matplotlib.pyplot as plt

x2 = -.18
x3 = .75  # these are initial guesses
def gdinexact(x2,x3):
    iterations = 1000 # number of iterations
    t = .5
    x = np.array([x2,x3])
    H = np.array([[10, 12],[12 ,20]])

    def s(x): # this function defines step, we are minimizing hence, - sign
        gd = -g(x) # step direction for gradient descent
        newt = -np.matmul(np.linalg.inv(H),g(x)).flatten() # step direction for newtons method
        return gd # return newt to use newtons method and gd to use gradient descent

    def d(x): # this is function being minimized
        d = (-2 + 2 * x[0] + 3 * x[1]) ** 2 + (0 - x[0]) ** 2 + (1 - x[1]) ** 2
        return d

    def g(x): # gradient of the function being minimized
        g = np.array([(10 * x[0] + 12 * x[1] - 8), (12 * x[0] + 20 * x[1] - 14)])
        return g

    def phi(a): # phi function being defined to compare function value at x-
        gx = g(x)
        gt = gx.transpose()
        st = s(x).transpose()
        phi = d(x) + (t * np.matmul(gt,st) * a) # step function
        return phi

    def linesearch(x,a):
        while d(x+a*s(x)) > phi(a): # -g(x) is step direction
            a = 0.5*a
        return a

    x_all = []
    f_all = []
    c_all = []
    fp_all = []
    a = 1
    #a = linesearch(x)
    fp = (np.absolute(d(x+a*s(x)) - d(x)))  # take the log of the difference between predicted and actual value
    fp_all.append(fp)
    c = 0
    c_all.append(c)
    x_all.append(x)
    f_all.append(d(x))
    #for i in range(iterations):
    while np.linalg.norm(g(x)) > 1e-6:
        a = linesearch(x,a)
        x = x+a*s(x)  #decrease x by multiplying current value and adding (alpha*step)
        x_all.append(x) # collect all x values in one array
        f_all.append(d(x)) # collect all function values in one array
        fp = (np.absolute(d(x+a*s(x))-d(x))) # take the log of the difference between predicted and actual value
        fp_all.append(fp)
        c = c+1
        c_all.append(c)

    plt.plot(c_all,fp_all)

    return x

print("final solution [x2,x3]=",(gdinexact(x2,x3)))
plt.yscale('log') # turn the y axis into the lognormal plot
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.suptitle('Initial value x2 = -.18 , x3 = .75')
plt.title('Lognormal Convergence plot')
plt.show()




