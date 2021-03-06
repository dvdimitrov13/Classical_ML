import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def fun(x, y):
    return (x ** 2) / 2 - (y ** 2) / 1.5


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Define the dimensions of the x and y axis
x = y = np.arange(-3.0, 3.0, 0.05)

# Create a rectangular grid with (x,y) coordinates for each point
X, Y = np.meshgrid(x, y)

# Calculte value of zs for each (x,y) pairing -- retuens a 1d array
zs = np.array([fun(x, y) for x, y in zip(np.ravel(X), np.ravel(Y))])

# Reshhape zs into a 'x' by 'y' matrix
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")

plt.show()
