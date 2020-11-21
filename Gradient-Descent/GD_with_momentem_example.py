import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
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
print( 'Solution found by scikit-learn : ', reg.coef_ )

# ------------------- plot data -------------------
X_line = np.linspace(0, 100, 1000)
X_line = np.c_[np.ones((1000, 1)), X_line]
y_line = reg.predict(X_line)
plt.plot(X, y, "b.")
plt.plot(X_line, y_line, "r-")
plt.xlabel("$Age$", fontsize=10)
plt.ylabel("$Price$", rotation=0, fontsize=10)
plt.show()





