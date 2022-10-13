# Problem 2
from bayes_opt import BayesianOptimization


def function(x, y):
   return ((4 - 2.1 * (x ** 2) + ((x ** 4) / 3)) * x ** 2 + (x * y) + (-4 + 4 * (y ** 2)) * (y ** 2))

pbounds = {'x': (-3, 3), 'y': (-2, 2)}

optimizer = BayesianOptimization(f=function, pbounds=pbounds, verbose=2, random_state=1)

optimizer.maximize(init_points=20, n_iter=0,)  # using several steps of random exploration to get a nice sample spread

print(optimizer.max)
print('Minimum function value is', function(-0.4848,  0.7409  ))

# from array of values, minimum is -.5211 at iteration 6 so this is assumed to be local solution since array is large