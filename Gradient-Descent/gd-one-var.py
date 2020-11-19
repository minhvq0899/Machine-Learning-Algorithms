"""
================================= Gradient Descent with one variable for Lin Reg =================================

All credit to Tiep Vu - the author of 'Machine Learning co ban'
I am replicating his code (with a few changes) for studying purpose only

"""

# -------------------------import-------------------------
import math
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation 

plt.style.use('seaborn-white')


# --------------------------------------compute GD--------------------------------------
def grad(x):
    return 2*x+ 5*np.cos(x)

def cost(x):
    return x**2 + 5*np.sin(x)

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1]) # x_updated = x mới được add vào list gần nhất - LR * đạo hàm của hàm số tại điểm x mới được add vào list gần nhất
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new) # để cuối cùng chúng ta có một list để visualization
    return (x, it)

# Sau khi có các hàm cần thiết, tôi thử tìm nghiệm với các điểm khởi tạo khác nhau là x0 = -5 và x0 = 5 
(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('For x0 = -5 and LR = .1, solution x1 = {0}, cost = {1}, obtained after {2} iterations'.format(x1[-1], cost(x1[-1]), it1))
print('For x0 = 5 and LR = .1, solution x2 = {0}, cost = {1}, obtained after {2} iterations'.format(x2[-1], cost(x2[-1]), it2))

# Tuy nhiên, với Learning Rate khác nhau thì có thể kết quả sẽ rất khác nhau
(x3, it3) = myGD1(.5, -5)
print('For x0 = -5 and LR = .5, solution x3 = {0}, cost = {1}, obtained after {2} iterations'.format(x3[-1], cost(x3[-1]), it3))


# ----------------------------------------Animation----------------------------------------
fig, ax = plt.subplots()

# compute x,y for cost function
x = np.asarray(x1)
y = cost(x)
print("x,y = {0}, {1}".format(x,y))
# compute 
x0 = np.linspace(-6, 6, 1000)
y0 = cost(x0)


# update function
def update(it):
    # initialize label
    label = 'Iteration: {0}                  Cost: {1}'.format(it, round(y[it], 3))
    # create an axis
    animlist = plt.cla() # clear an existed axis
    animlist = plt.axis()
    animlist = plt.axis([-6, 6, -10, 35])
    animlist = plt.plot(x0, y0)
    # update axis for animation
    if it == 0:
    	animlist = plt.plot(x[it], y[it], 'ro', markersize = 7)
    else:
        if it >= len(x): 
            pass
        else: 
            animlist = plt.plot(x[it-1:it+1], y[it-1:it+1], 'ro', markersize = 7)
            animlist = plt.plot([x[it-1], x[it]], [y[it-1], y[it]], 'k-')
    	
    ax.set_xlabel(label)

    return animlist, ax

# create an animation plot
anim = FuncAnimation(fig, update, frames=np.arange(0, len(x)), interval=400)
plt.show()



# the update function is a little bit extra, but I did that to learn more about 
# matplotlib.animation.FuncAnimation. 