import sys

sys.path.append("C:\\Users\\minhv\\Machine-Learning-Algorithms\\linreg")

from multilinear_Normal_Equation import *
from sklearn.linear_model import LinearRegression

# fit the model by Linear Regression
reg = LinearRegression(fit_intercept=False) # fit_intercept = False for calculating the bias
reg.fit(Xbar, y)

# Compare two results
print( 'Solution found by scikit-learn  : ', reg.coef_ )
print( '\nSolution found by (5): ', w.T)
