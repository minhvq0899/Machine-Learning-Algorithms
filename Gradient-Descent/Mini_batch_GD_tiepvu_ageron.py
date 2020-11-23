"""
========================================= Minibatch Gradient Descent =========================================
        
All credit to Tiep Vu (the author of 'Machine Learning co ban') and Aurélien Geron (the author of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)
I am replicating their code (with a few changes) for studying purpose only

Content of this code:
1. Toy Dataset (y = 4 + 3x + noise)
2. Minibatch Gradient Descent (SGD)
3. Animation
4. Example from Boston dataset

""" 

import numpy as np 
np.set_printoptions(suppress=True)
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression


# ----------------------------------------------------------------- toy data -----------------------------------------------------------------
np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1)

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)
# print("Number of data points: ", Xbar.shape[0], "\n")

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_exact = np.dot(np.linalg.pinv(A), b)
print("w_exact using Normal Equation: ", w_exact, "\n")


# ----------------------------------------------- Minibatch Gradient Descent (SGD) -----------------------------------------------
# gradient descent using the whole dataset
def grad(w):
	return 1/Xbar.shape[0] * Xbar.T.dot(Xbar.dot(w) - y)

# We will have to compute gradient at just a batch of random points --> need new grad() function
def minibatch_grad(w, minibatch_size, i, rd_id):
    Xbar_shuffled = Xbar[rd_id] 
    y_shuffled = y[rd_id]
    # choose 1 batch 
    xi = Xbar_shuffled[i:i+minibatch_size] 
    yi = y_shuffled[i:i+minibatch_size]
    g = 2/minibatch_size * xi.T.dot(xi.dot(w) - yi)
    return g

# SGD from Aurélien Geron's
def learning_schedule(t):
    t0, t1 = 200, 1000  # learning schedule hyperparameters
    return t0 / (t + t1)
     
def minibatchGD_ageron(theta_init, minibatch_size, n_epochs, minibatch_grad):
    theta = [theta_init]
    eta_ageron = [learning_schedule(0)]
    theta_last_check = theta_init
    N = Xbar.shape[0]
    count = 0
    for epoch in range(n_epochs):
        rd_id = np.random.permutation(N) 
        for i in range(0, N, minibatch_size):
            count += 1     
            g = minibatch_grad(theta[-1], minibatch_size, i, rd_id)
            eta_current = learning_schedule(count) # this means eta (learning rate) will gradually get smaller
            eta_ageron.append(eta_current)
            theta_new = theta[-1] - eta_current * g
            theta.append(theta_new)
            if count % 5 == 0: # check after every 5 iterations
                theta_this_check = theta_new                 
                if np.linalg.norm(theta_this_check - theta_last_check)/len(theta_init) < 1e-3:                                    
                    print("Early Stopping for Ageron\n")
                    return (theta, eta_ageron, count) # number of iterations 
                theta_last_check = theta_this_check
    
    return (theta, eta_ageron, count)



# ------------------------------------------------------------- Animation -------------------------------------------------------------
# Contour
N = X.shape[0]
a1 = np.linalg.norm(y, 2)**2/N
b1 = 2*np.sum(X)/N
c1 = np.linalg.norm(X, 2)**2/N
d1 = -2*np.sum(y)/N 
e1 = -2*X.T.dot(y)/N

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
xg = np.arange(1.5, 7.0, delta)
yg = np.arange(0.5, 4.5, delta)
Xg, Yg = np.meshgrid(xg, yg)
Z = a1 + Xg**2 +b1*Xg*Yg + c1*Yg**2 + d1*Xg + e1*Yg

# ------------------------------ Animation for Aurélien Geron's code ------------------------------
# Animation Plot
# Compute theta using Aurélien Geron's minibatch GD
theta_init = np.array([[2], [1]])
(theta_ageron, eta_ageron, it_ageron) = minibatchGD_ageron(theta_init, 5, 10, minibatch_grad)
print("theta_ageron using minibatchGD_ageron: {0}, after {1} iterations and achieve norm 2 of gradient at {2}".format(theta_ageron[-1].T, it_ageron, round(np.linalg.norm(grad(theta_ageron[-1])), 3))) 

def update_ageron(ii):
    if ii == 0:
        plt.cla()
        CS = plt.contour(Xg, Yg, Z, 100)
        manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3)]
        animlist = plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
        plt.plot(w_exact[0], w_exact[1], 'go')
    else:
        animlist = plt.plot([theta_ageron[(ii-1)][0], theta_ageron[ii][0]], 
                            [theta_ageron[(ii-1)][1], theta_ageron[ii][1]], 'r-')

    animlist = plt.plot(theta_ageron[ii][0], theta_ageron[ii][1], 'ro', markersize = 4) 
    xlabel = '$\eta =$ %.3f' % eta_ageron[ii] + '; iter = %d/%d' %(ii, it_ageron)
    xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(grad(theta_ageron[ii]))
    ax.set_xlabel(xlabel)
    animlist = plt.title('LR with minibatch GD Aurélien Geron')
    return animlist, ax

fig, ax = plt.subplots(figsize=(4,4))    
plt.cla()
plt.axis([1.5, 7, 0.5, 4.5])

anim1 = FuncAnimation(fig, update_ageron, frames=np.arange(0, it_ageron), interval= 100)
plt.show() 







'''
# --------------------------------------------- Example from Boston Dataset  ----------------------------------------------
boston = load_boston()

X = pd.DataFrame(boston.data, columns = boston.feature_names)
# we are only doing one variable linear regression
y = pd.DataFrame(boston.target).to_numpy()

# --------------------------- compute exact w ---------------------------
# Building Xbar 
# We need to add a column of 1s in front of X so the final result will yield another element, which is theta 0
one = np.ones((X.shape[0], 1)) 
Xbar = np.concatenate((one, X), axis = 1) 

# Scikit library
reg = LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
reg.fit(Xbar, y)
print('Solution found by scikit-learn : ', reg.coef_, '\n')


# ---------------------------- Use Aurélien Geron's GD ----------------------------
# Compute theta using Aurélien Geron's minibatch GD
theta_init = np.array([[36], [-0.1], [0.1], [0.1], [2], [-16], [4], [0.1], [-2], [0.5], 
                       [-0.1], [-1], [0.01], [-0.7]])

# before run this line, please adjust the learning rate t0 above
(theta_ageron, eta_ageron, it_ageron) = minibatchGD_ageron(theta_init, 5, 10, minibatch_grad)
print("theta_ageron using minibatchGD_ageron: {0}, after {1} iterations and achieve norm 2 of gradient at {2}".format(theta_ageron[-1].T, it_ageron, round(np.linalg.norm(grad(theta_ageron[-1])), 3))) 
# norm_ageron = np.linalg.norm(grad(theta_ageron))    
# print(theta_ageron[0:100])
'''





