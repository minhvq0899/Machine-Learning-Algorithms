"""
Author: Minh Q. Vu

================================================== Softmax Regression ==================================================
        
All credit to Tiep Vu (the author of 'Machine Learning co ban') and Aurélien Geron (the author of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)
I am replicating their code (with a few changes) and adding comments along the way for a studying purpose only

Content of this code:
1.  Toy data

2.  One-hot Encoding

3.  Functions
    softmax(Z)
    softmax_stable(Z)
    cost(X, Y, W)
    grad(X, Y, W)
    numerical_grad(X, Y, W, cost)
    softmax_regression(X, y, W_init, eta, tol = 1e-4, max_count = 10000)
    pred(W, X)
    plot_clusters(X, ylabel = None)

4. Simulated Data

5. Run algorithms and plot decision boundaries

"""
import numpy as np 
import matplotlib.pyplot as plt

# ======================================================= Toy Data =======================================================
N = 2 # number of training sample 
d = 2 # data dimension 
C = 3 # number of classes 

X = np.random.randn(d, N)
y = np.random.randint(0, 3, (N,))
print("X: ", X, "\n")
print("y: ", y, "\n")










# =================================================== One-hot Encoding ===================================================
# We need to transform vector label yi to one-hot encoding form Yi vector. With N data points and C classes, we have a 
# matrix C x N. If we store this in RAM, it will be a waste of memory. Solution: sparse matrix
from scipy import sparse 
def convert_labels(y, C = C):
    """
    convert 1d label to a matrix label: each column of this matrix coresponding to 1 element in y. In i-th column of Y, 
    only one non-zeros element located in the y[i]-th position, and = 1 
    Ex: y = [0, 2, 1, 0], and 3 classes then return

            [[1, 0, 0, 1],
             [0, 0, 1, 0],
             [0, 1, 0, 0]]
    """
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape = (C, len(y))).toarray()
    return Y 

Y = convert_labels(y, C)
print("Y: ", Y, "\n")












# ====================================================== Functions ======================================================
"""
The two versions of Softmax functions
"""
def softmax(Z):
    # Compute softmax values for each sets of scores in V. Each column of V is a set of score.    
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis = 0)
    return A

def softmax_stable(Z):
    # Compute softmax values for each sets of scores in Z. Each column of Z is a set of score.    
    e_Z = np.exp(Z - np.max(Z, axis = 0, keepdims = True))
    A = e_Z / e_Z.sum(axis = 0)
    return A

"""
Most important thing in optimizing loss function is computing gradient. In real life, solution is compared to gradient 
computed with numeric gradient. 
"""
# cost or loss function  
def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y*np.log(A))

W_init = np.random.randn(d, C)

def grad(X, Y, W):
    A = softmax((W.T.dot(X)))
    E = A - Y
    return X.dot(E.T)
    
def numerical_grad(X, Y, W, cost):
    eps = 1e-6
    g = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_p = W.copy()
            W_n = W.copy()
            W_p[i, j] += eps 
            W_n[i, j] -= eps
            g[i,j] = (cost(X, Y, W_p) - cost(X, Y, W_n))/(2*eps)
    return g 


g1 = grad(X, Y, W_init)
g2 = numerical_grad(X, Y, W_init, cost)

print("Checking gradient... ", np.linalg.norm(g1 - g2), "\n") # if this value is small enough (< 1e-6) then the gradient computed is acceptable


# --------------------- Main functions ---------------------
def softmax_regression(X, y, W_init, eta, tol = 1e-4, max_count = 10000):
    W = [W_init]    
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data 
        mix_id = np.random.permutation(N)
        # go through each data points in the permutation
        for i in mix_id:
            # take 1 instance and its label out 
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            # first compute the logit y, the input that logit y in a softmax function
            # output will be probability add up to 1
            ai = softmax(np.dot(W[-1].T, xi))
            # update w
            W_new = W[-1] + eta*xi.dot((yi - ai).T)
            count += 1
            # stopping criteria
            if count%check_w_after == 0:                
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W

eta = .05 
d = X.shape[0]
W_init = np.random.randn(d, C)

W = softmax_regression(X, y, W_init, eta)
# W[-1] is the solution, W is all history of weights


# 
def pred(W, X):
    """
    predict output of each columns of X
    Class of each x_i is determined by location of max probability
    Note that class are indexed by [0, 1, 2, ...., C-1]
    """
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis = 0)


def plot_clusters(X, ylabel = None):
    plt.figure(figsize=(12, 6))
    plt.scatter(X[:, 1], X[:, 2], c=ylabel, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$x_2$", fontsize=14, rotation=0)
    plt.show()


# ======================================================= Simulated Data =======================================================
# Create 3 clusters of data points
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

# each column is a datapoint
X = np.concatenate((X0, X1, X2), axis = 0).T 
# extended data
X = np.concatenate((np.ones((1, 3*N)), X), axis = 0)
C = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

print("X: ", X, "\nwith shape: ", str(X.shape), "\n")
print("Labels: ", original_label, "\n")















# ======================================================= Run algorithms =======================================================
plot_clusters(X.T, original_label)

W_init = np.random.randn(X.shape[0], C) # X.shape[0] is number of dimension + 1, C is number of classes
print("W_init: ", W_init, "\n")

W = softmax_regression(X, original_label, W_init, eta)
print(W[-1])

# ------------------------- decision boundary -------------------------
xm = np.arange(-2, 11, 0.025)
xlen = len(xm)
ym = np.arange(-3, 10, 0.025)
ylen = len(ym)
xx, yy = np.meshgrid(xm, ym)
xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)

XX = np.concatenate((np.ones((1, xx.size)), xx1, yy1), axis = 0)

print("XX's shape: ", XX.shape)
Z = pred(W[-1], XX)
print("----- Predicting Z = pred(W[-1], XX) ... -----")
print("Z's shape: ", Z.shape, "\n")

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 6))
plt.contourf(xx, yy, Z, 200, cmap='jet', alpha = .1)
plt.scatter(X.T[:, 1], X.T[:, 2], c=original_label, s=1)
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plt.show()

















