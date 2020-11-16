from load-boston import *
import numpy as np

# Building Xbar 
one = np.ones((X.shape[0], 1)) # 506 rows, 1 column of 1s
Xbar = np.concatenate((one, X), axis = 1) # 506 columns, 14 rows

print("\nXbar:\n ", Xbar)
print("\nXbar's shape: ", Xbar.shape)

# Calculating weights of the fitting line 
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
print('w = ', w)


