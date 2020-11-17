import math
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.datasets import load_boston

# boston = load_boston()

# X = pd.DataFrame(boston.data, columns = boston.feature_names)
# y = pd.DataFrame(boston.target)

X = np.random.rand(5, 1)
y = 4 + 3 * X + .2*np.random.randn(5, 1) # noise added

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

print(one)
print(Xbar)
print(Xbar.shape)
print(Xbar.T)


