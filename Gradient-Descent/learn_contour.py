import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
np.set_printoptions(suppress=True)

# ----------------------------------------- learn meshgrid -----------------------------------------
"""
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
X, Y = np.meshgrid(x, y, sparse=True)
z = np.sin(X**2 + Y**2) / (X**2 + Y**2)
h = plt.contourf(X,Y,z)
plt.show()
"""
# ----------------------------------------- learn contour -----------------------------------------
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)

x = np.linspace(0, 5, 50)
y = np.linspace(0, 5, 40)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, 20, cmap='RdGy')
plt.show()

