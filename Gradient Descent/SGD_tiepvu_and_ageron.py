"""
========================================= Stochastic Gradient Descent =========================================
        
All credit to Tiep Vu (the author of 'Machine Learning co ban') and Aurélien Geron (the author of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)
I am replicating their code (with a few changes) for studying purpose only

Content of this code:
1. Stochastic Gradient Descent (SGD)
    1.1. Tiep Vu's code
    1.2. Aurélien Geron's code

2. Animation
    2.1. Tiep Vu's animation
    2.2. Aurélien Geron's animation
"""

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------- toy data -----------------------------------------------------------------
np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1)

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)
print("Number of data points: ", Xbar.shape[0], "\n")

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_exact = np.dot(np.linalg.pinv(A), b)
print("w_exact using Normal Equation: ", w_exact, "\n")


# ----------------------------------------------- Stochastic Gradient Descent (SGD) -----------------------------------------------
# Mỗi lần duyệt một lượt qua TẤT CẢ các điểm trên toàn bộ dữ liệu được gọi là một epoch

def grad(w):
	return 1/Xbar.shape[0] * Xbar.T.dot(Xbar.dot(w) - y)

# We will have to compute gradient at just ONE random point --> need new grad() function
def sgrad(w,i,rd_id):
    true_i = rd_id[i]
    # choose 1 point at random
    xi = Xbar[true_i, :] 
    yi = y[true_i]
    residual = np.dot(xi, w) - yi
    return (xi*residual).reshape(2,1)

# SGD function from Tiep Vu
def SGD_tiepvu(w_init, n_epochs, eta, sgrad):
    w = [w_init]
    w_last_check = w_init
    iter_check_w = 10
    N = X.shape[0]
    count = 0
    # this means we will go through the whole dataset (randomly) n_epochs times
    for epoch in range(n_epochs):
        # shuffle data -> rd_id will be a numpy array of N item, ordering randomly
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1
            g = sgrad(w[-1], i, rd_id)
            w_new = w[-1] - eta*g
            w.append(w_new)
            if count % iter_check_w == 0: # check after every 10 iterations
                w_this_check = w_new                 
                if np.linalg.norm(w_this_check - w_last_check)/len(w_init) < 1e-3:                                    
                    print("Early Stopping for Tiep Vu\n")
                    return (w, count) # number of iterations 
                w_last_check = w_this_check
    
    return (w, count)


# SGD from Aurélien Geron's
def learning_schedule(t):
    t0, t1 = 50, 150  # learning schedule hyperparameters
    return t0 / (t + t1)
     
def SGD_ageron(theta_init, n_epochs, sgrad):
    theta = [theta_init]
    eta_ageron = [learning_schedule(0)]
    theta_last_check = theta_init
    N = Xbar.shape[0]
    count = 0
    for epoch in range(n_epochs):
        rd_id = np.random.permutation(N)
        for i in range(N):
            count += 1     
            g = sgrad(theta[-1], i, rd_id)
            eta_current = learning_schedule(count) # this means eta (learning rate) will gradually get smaller
            eta_ageron.append(eta_current)
            theta_new = theta[-1] - eta_current * g
            theta.append(theta_new)
            if count % 10 == 0: # check after every 10 iterations
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

theta_init, w_init = np.array([[2], [1]]), np.array([[2], [1]])

# ------------------------------ Animation for Tiep Vu's code ------------------------------
# Animation Plot
# Compute theta using Tiep's SGD
eta = 0.1
(w_tiep, it_tiep) = SGD_tiepvu(w_init, 10, eta, sgrad)
print("w_tiep using SGD_tiepvu: {0}, after {1} iterations and achieve norm 2 of gradient at {2} \n".format(w_tiep[-1].T, it_tiep, round(np.linalg.norm(grad(w_tiep[-1])), 3)))
# print("Gradient is small enough (<1e-3) at: ", np.linalg.norm(grad(w_tiep[-1])))  

def update_tiep(ii):
    if ii == 0:
        plt.cla()
        CS = plt.contour(Xg, Yg, Z, 100)
        manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3)]
        animlist = plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
        plt.plot(w_exact[0], w_exact[1], 'go')
    else:
        animlist = plt.plot([w_tiep[(ii-1)][0], w_tiep[ii][0]], 
                            [w_tiep[(ii-1)][1], w_tiep[ii][1]], 'r-')

    animlist = plt.plot(w_tiep[ii][0], w_tiep[ii][1], 'ro', markersize = 4) 
    xlabel = '$\eta =$ ' + str(eta) + '; iter = %d/%d' %(ii, it_tiep)
    xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(grad(w_tiep[ii]))
    ax.set_xlabel(xlabel)
    animlist = plt.title('LR with SGD Tiep Vu')
    return animlist, ax

fig, ax = plt.subplots(figsize=(4,4))    
plt.cla()
plt.axis([1.5, 7, 0.5, 4.5])

anim1 = FuncAnimation(fig, update_tiep, frames=np.arange(0, it_tiep), interval= 100)
plt.show() 


# ------------------------------ Animation for Aurélien Gerons code ------------------------------
# Animation Plot
# Compute theta using Aurélien Geron's SGD
(theta_ageron, eta_ageron, it_ageron) = SGD_ageron(theta_init, 10, sgrad)
print("theta_ageron using SGD_ageron: {0}, after {1} iterations and achieve norm 2 of gradient at {2}\n    ".format(theta_ageron[-1].T, it_ageron, round(np.linalg.norm(grad(theta_ageron[-1])), 3)))
# print("Gradient is small enough (<1e-3) at: ", np.linalg.norm(grad(theta_ageron[-1])))  

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
    animlist = plt.title('LR with SGD Ageron')
    return animlist, ax

fig, ax = plt.subplots(figsize=(4,4))    
plt.cla()
plt.axis([1.5, 7, 0.5, 4.5])

anim1 = FuncAnimation(fig, update_ageron, frames=np.arange(0, it_ageron), interval=100)
plt.show() 









