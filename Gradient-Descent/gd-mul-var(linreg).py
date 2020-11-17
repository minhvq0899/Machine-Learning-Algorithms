import math
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------load dataset------------------------------------
from sklearn.datasets import load_boston

# load dataset
boston = load_boston()

X = pd.DataFrame(boston.data, columns = boston.feature_names)
y = pd.DataFrame(boston.target)

# Building Xbar 
# We need to add a column of 1s in front of X so the final result will yield another element, which is theta 0
one = np.ones((X.shape[0], 1)) # 506 rows, 1 column of 1s
Xbar = np.concatenate((one, X), axis = 1) # 506 columns, 14 rows

# ------------------function to compute derivate and cost of linear regression's cost function------------------
def grad(w):
    N = Xbar.shape[0] # number of rows/ instances/ observations
    return 1/N * Xbar.T.dot(Xbar.dot(w) - y)

def cost(w):
    N = Xbar.shape[0] # number of rows/ instances/ observations
    return .5/N*np.linalg.norm(y - Xbar.dot(w), 2)**2

def myGD(w_init, grad, eta):
    w = [w_init]
    for it in range(1000):
        w_new = w[-1] - eta*grad(w[-1])
        if np.linalg.norm(grad(w_new))/len(w_new) < 1e-3:
            break 
        w.append(w_new)
    return (w, it) 


# --------------------------------------compute w using gradient-------------------------------------------
w_init = np.array([[2], [1]])
(w1, it1) = myGD(w_init, grad, 0.1)
print('Solution found by GD: w = ', w1[-1].T, ',\nafter %d iterations.' %(it1+1))



