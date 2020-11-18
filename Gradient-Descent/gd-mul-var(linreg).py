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


def myGD(w_init, n_iterations, eta, grad):
    w = [w_init]
    for it in range(n_iterations):
        w_new = w[-1] - eta*grad(w[-1])
        # đến khi nào trung bình cộng của cả 14 gradient là đủ nhỏ thì ta dùng lại 
        norm = np.linalg.norm(grad(w_new))/len(w_new)
        print(norm)
        if norm < 1e-3: 
            break 
        w.append(w_new)
    return (w, it) 


def GD(w_init, n_iterations, eta, grad):
    for it in range(n_iterations):
        gradients = grad(w)
        w = w - eta * gradients
    
    return w


# --------------------------------------compute w using gradient-------------------------------------------
w_init = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]) # 14 rows, 1 column 

# print("w_init: ", w_init, ", type is ", str(type(w_init)), ", shape is ", str(w_init.shape))
# print("[w_init]: ", [w_init], "type is ", str(type([w_init])))

(w_final, it_final) = myGD(w_init, 20000, 0.000005, grad)

print('Solution found by GD: w = ', w_final[-1].T, ',\nafter %d iterations.' %(it_final+1))



