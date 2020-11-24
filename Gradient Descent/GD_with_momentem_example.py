"""
Author: Minh Vu

Content: 
1. Load dataset and compute the exact theta using Normal Equation
2. Use GD with Momentum to learn
3. Contour
4. 

"""
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import math

# --------------------------------------------- load dataset ---------------------------------------------
boston = load_boston()

X = pd.DataFrame(boston.data, columns = boston.feature_names)
# we are only doing one variable linear regression
# PRICE = theta0 + theta1 * AGE
X = X['AGE'].to_frame()
y = pd.DataFrame(boston.target)

# ------------- compute exact w -------------
# Building Xbar 
# We need to add a column of 1s in front of X so the final result will yield another element, which is theta 0
one = np.ones((X.shape[0], 1)) 
Xbar = np.concatenate((one, X), axis = 1) 

# Normal Equation
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
theta_exact = np.dot(np.linalg.pinv(A), b)
print('theta_exact = ', theta_exact)

# Scikit library
reg = LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
reg.fit(Xbar, y)
print( '\nSolution found by scikit-learn : ', reg.coef_ )

# ------------------- plot data -------------------
"""
X_line = np.linspace(0, 100, 1000)
X_line = np.c_[np.ones((1000, 1)), X_line]
y_line = reg.predict(X_line)
plt.plot(X, y, "b.")
plt.plot(X_line, y_line, "r-")
plt.xlabel("$Age$", fontsize=10)
plt.ylabel("$Price$", rotation=0, fontsize=10)
plt.show()
"""
# --------------------------------------- use GD with Momentum to learn ---------------------------------------
def cost(theta):
	return .5/Xbar.shape[0]*np.linalg.norm(y - Xbar.dot(theta), 2)**2

def grad(theta):
	return 1/Xbar.shape[0] * Xbar.T.dot(Xbar.dot(theta) - y)

def GD_momentum(theta_init, grad, eta, gamma):
    theta = [theta_init]
    v = [np.zeros_like(theta_init)]
    for it in range(100):
        v_new = gamma*v[-1] + eta*grad(theta[-1])
        theta_new = theta[-1] - v_new
        # check for stopping criteria
        # stop when the gradient at new theta is small enough
        grad_check = np.linalg.norm(grad(theta_new))/len(theta_new)
        # print(grad_check)
        if grad_check < 1e-3:
            break
        
        theta.append(theta_new)
        v.append(v_new)

    return (theta, it)

# compute
theta_init = np.array([[28.5], [1]])
(theta_mm, it_mm) = GD_momentum(theta_init, grad, .0005, .9)

print("\nSolution by GD with Momentum: ", theta_mm)


# ----------------------------------------------------- Contour ----------------------------------------------------- 
N = X.shape[0]
print(X.shape, N)
a1 = np.linalg.norm(y, 2)**2/N
b1 = 2*np.sum(X)/N
c1 = np.linalg.norm(X, 2)**2/N
d1 = -2*np.sum(y)/N 
e1 = -2*X.T.dot(y)/N

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

delta = 0.025
xg = np.arange(27.5, 33, delta)
yg = np.arange(-2, 2, delta)
Xg, Yg = np.meshgrid(xg, yg)
Z = a1 + Xg**2 +b1*Xg*Yg + c1*Yg**2 + d1*Xg + e1*Yg


# ----------------------------------------------------- Animation -----------------------------------------------------
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def update_mm(ii):
    if ii == 0:
        plt.cla()
        CS = plt.contour(Xg, Yg, Z, 100)
        # manual_locations = [(4.5, 3.5), (4.2, 3), (4.3, 3.3)]
        # animlist = plt.clabel(CS, inline=.1, fontsize=10, manual=manual_locations)
        plt.plot(theta_exact[0], theta_exact[1], 'go')
    else:
        animlist = plt.plot([theta_mm[ii-1][0], theta_mm[ii][0]], [theta_mm[ii-1][1], theta_mm[ii][1]], 'r-')
    
    animlist = plt.plot(theta_mm[ii][0], theta_mm[ii][1], 'ro', markersize = 4) 
    xlabel = '$\eta =$ ' + str(0.0005) + '; iter = %d/%d' %(ii, it_mm)
    xlabel += '; ||grad||_2 = %.3f' % np.linalg.norm(grad(theta_mm[ii]))
    ax.set_xlabel(xlabel)
    
    return animlist, ax
    

# ----------- visualization -------------
fig, ax = plt.subplots(figsize=(4,4))    
plt.cla()
plt.axis([23, 27, -1, 1])

anim1 = FuncAnimation(fig, update_mm, frames=np.arange(0, it_mm), interval=200)
plt.show()




















