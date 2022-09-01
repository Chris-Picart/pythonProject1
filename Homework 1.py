from scipy.optimize import minimize

x0 = [7, 4, 6, 8, 9]  # this is the initial guess for each x
fun = lambda x: (x[0] - x[1]) ** 2 + (x[1] + x[2] - 2) ** 2 + (x[3] - 1) ** 2 + (
        x[4] - 1) ** 2  # function to be minimized
cons = ({'type': 'eq', 'fun': lambda x: x[0] + 3 * x[1]},  # equality constraints
        {'type': 'eq', 'fun': lambda x: x[2] + x[3] - 2 * x[4]},  # equality constraints
        {'type': 'eq', 'fun': lambda x: x[1] - x[4]})  # equality constraints
bnds = ((-10, 10), (-10, 10), (-10, 10), (-10, 10), (-10, 10))  # bounds for ( x1, x1,....x5)
res = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)  # use minimize function with required inputs

print(res)  # Show results

# the minimized value of the function did not change when the initial guess values were varied
#  minimized function value = 4.093023255814588
