"""
Author: Minh Q. Vu

================================================== Logistic Regression ==================================================
        
All credit to Tiep Vu (the author of 'Machine Learning co ban') and Aurélien Geron (the author of Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow)
I am replicating their code (with a few changes) and adding comments along the way for a studying purpose only

Content of this code:
1. Toy data
    1.1.    1D data
    1.2.    2D data
2. Functions for logistic regression numpy
    2.1.    sigmoid()
    2.2     grad_logistic_reg()
    2.3     logistic_sigmoid_regression()
3. Main function
    3.1.    find w for 1D data
    3.2.    plot for 1D data
    3.3.    find w for 2D data
    3.4.    plot for 2D data

""" 

import numpy as np  
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
np.random.seed(2)

# ====================================================== Toy data ======================================================
# 1D data
X_1d = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 
              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y_1d = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

X_1d = np.concatenate((np.ones((1, X_1d.shape[1])), X_1d), axis = 0)

# 2D data
from sklearn import datasets
iris = datasets.load_iris()

X_2d = iris["data"][:, (2, 3)]  # petal length, petal width
y_2d = (iris["target"] == 2).astype(np.int)  # 1 if Iris virginica, else 0

X_2d = np.concatenate((np.ones((X_2d.shape[0], 1)), X_2d), axis = 1)












# ============================================ Functions for logistic regression numpy ============================================
# sigmoid function
def sigmoid(s):
    return 1/(1 + np.exp(-s))

# derivative of the Logistic Regression's cost function with respect to w at a single training instance 
def grad_logistic_reg(yi, zi, xi):
    return (zi - yi)*xi


# use SGD to find w for the logistic regression
def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]    
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data 
        rd_id = np.random.permutation(N)
        for i in rd_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] - eta * grad_logistic_reg(yi, zi, xi)
            count += 1
            # stopping criteria
            if count % check_w_after == 0:                
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)

    return (w, count)














# ==================================================== Main function ====================================================
if __name__ == "__main__":
    
    # ------------------------ find w for 1D data ------------------------
    #print("X_1d is {0} with shape {1} \n".format(X_1d, X_1d.shape))
    eta = .05 
    d = X_1d.shape[0]
    w_init = np.random.randn(d, 1)

    (w_1d, count) = logistic_sigmoid_regression(X_1d, y_1d, w_init, eta)
    print( "After {0} iterations, we found w as {1} \n".format(count, w_1d[-1]) )
    print( "This means output y can be predicted as y = sigmoid( %.2f + %.2f * x ) \n" % ( w_1d[-1][0], w_1d[-1][1] ) )


    # ------------------------ plot for 1D data ------------------------
    X0 = X_1d[1, np.where(y_1d == 0)][0]
    y0 = y_1d[np.where(y_1d == 0)]
    X1 = X_1d[1, np.where(y_1d == 1)][0]
    y1 = y_1d[np.where(y_1d == 1)]

    plt.plot(X0, y0, 'ro', markersize = 8)
    plt.plot(X1, y1, 'bs', markersize = 8)

    xx = np.linspace(0, 6, 1000)
    w0 = w_1d[-1][0][0]
    w1 = w_1d[-1][1][0]
    threshold = -w0/w1
    yy = sigmoid(w0 + w1*xx)
    plt.axis([-2, 8, -1, 2])
    plt.plot(xx, yy, 'g-', linewidth = 2)
    plt.plot(threshold, .5, 'y^', markersize = 8)
    plt.xlabel('studying hours')
    plt.ylabel('predicted probability of pass')
    plt.show()
    
    









    # --------------------------------------- find w for 2d data ---------------------------------------
    # print("X_2d: \n", X_2d)
    X_2d = X_2d.T
    # print("X_2d: \n", X_2d)
    # print(y_2d)

    eta = 0.05
    d = X_2d.shape[0]
    w_init = np.random.randn(d, 1)

    (w_2d, count) = logistic_sigmoid_regression(X_2d, y_2d, w_init, eta)
    print( "After {0} iterations, we found w as {1} \n".format(count, w_2d[-1]) )
    print( "This means output y can be predicted as y = sigmoid( %.2f + %.2f * x1 + %.2f * x2 ) \n" % ( w_2d[-1][0], w_2d[-1][1], w_2d[-1][2] ) )
    # print( "Predicted y: ", sigmoid(np.dot(w_2d[-1].T, X_2d)) )
    


    # ------------------------------------- plot for 2d data -------------------------------------
    x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]
    # print("X_new shape: ", X_new.shape)
    X_new = np.concatenate((np.ones((X_new.shape[0], 1)), X_new), axis = 1)
    X_new = X_new.T

    y_proba = sigmoid(np.dot(w_2d[-1].T, X_new))

    plt.figure(figsize=(10, 4))
    plt.plot(X_2d[1, y_2d==0], X_2d[2, y_2d==0], "bs", label = "Not Iris virginica")
    plt.plot(X_2d[1, y_2d==1], X_2d[2, y_2d==1], "g^", label = "Iris virginica")

    zz = y_proba.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#fafab0','#9898ff'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)

    # left_right = np.array([2.9, 7])
    # boundary = -(w_2d[-1][0][0] * left_right + w_2d[-1][0]) / w_2d[-1][0][1]

    plt.clabel(contour, inline=1, fontsize=12)
    # plt.plot(left_right, boundary, "k--", linewidth=3)
    # plt.text(3.5, 1.5, "Not Iris virginica", fontsize=14, color="b", ha="center")
    # plt.text(6.5, 2.3, "Iris virginica", fontsize=14, color="g", ha="center")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.axis([2.9, 7, 0.8, 2.7])
    plt.show()


