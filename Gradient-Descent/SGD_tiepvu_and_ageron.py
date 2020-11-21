"""
========================================= Different Varients of Gradient Descent =========================================
        
All credit to Tiep Vu - the author of 'Machine Learning co ban'
I am replicating his code (with a few changes) for studying purpose only

Content of this code:
1. Stochastic Gradient Descent (SGD)
    1.1. Tiep Vu's code
    1.2. Aurélien Geron's code

2. Animation

"""

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# ---------------------------------------------- toy data ----------------------------------------------
np.random.seed(2)

X = np.random.rand(1000, 1)
y = 4 + 3 * X + .2*np.random.randn(1000, 1)

# Building Xbar 
one = np.ones((X.shape[0],1))
Xbar = np.concatenate((one, X), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w_exact = np.dot(np.linalg.pinv(A), b)
print("w_exact using Normal Equation: ", w_exact)
# ------------------------------------------ Stochastic Gradient Descent (SGD) ------------------------------------------
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
                    return (w, epoch*N + i) # number of iterations 
                w_last_check = w_this_check
    return (w, n_epochs*N)

"""
# SGD from Aurélien Geron's
def learning_schedule(t):
    return t0 / (t + t1)
     
def SGD_ageron():
    theta = np.random.randn(2,1)  # random initialization

    for epoch in range(n_epochs):
        for i in range(m):
            if epoch == 0 and i < 20:                    
                y_predict = X_new_b.dot(theta)           
                style = "b-" if i > 0 else "r--"         
                plt.plot(X_new, y_predict, style)        
            random_index = np.random.randint(m)
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients
            theta_path_sgd.append(theta)                 
"""


# ------------------------------------------------ Animation ------------------------------------------------
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

# -------------- Animation for Tiep Vu's code --------------
# Animation Plot
# Compute theta using Tiep's SGD
w_init = np.array([[2], [1]])
eta = 0.1
w_tiep = SGD_tiepvu(w_init, 10, 0.1, sgrad)
print("w_tiep using SGD_tiepvu: ", w_tiep[-1])

def update_tiep(ii):
    if ii == 0:
        plt.cla()
        CS = plt.contour(Xg, Yg, Z, 100)
        manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3)]
        animlist = plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
        plt.plot(w_exact[0], w_exact[1], 'go')
    else:
        animlist = plt.plot([w_tiep[(ii-1)*batch_it][0], w_tiep[ii*batch_it][0]], 
                            [w_tiep[(ii-1)*batch_it][1], w_tiep[ii*batch_it][1]], 'r-')
    animlist = plt.plot(w_tiep[ii*batch_it][0], w_tiep[ii*batch_it][1], 'ro', markersize = 4) 
    xlabel = '$\eta =$ ' + str(eta) + '; iter = %d/%d' %(ii*batch_it, it)
    xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(grad(w_tiep[ii*batch_it]))
    ax.set_xlabel(xlabel)
    animlist = plt.title('LR with SGD')
    return animlist, ax


batch_it = 20
it = len(w_tiep)
fig, ax = plt.subplots(figsize=(4,4))    
plt.cla()
plt.axis([1.5, 7, 0.5, 4.5])

anim1 = FuncAnimation(fig, update_tiep, frames=np.arange(0, it//batch_it), interval=200)
plt.show() 










