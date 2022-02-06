import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import random


def rosenbrock(x, a=1, b=100):
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def quadratic_f(x):
    return x[0] ** 2 + x[1] ** 2


# Analytic gradient
def gRosenbrock(f, x, a=1, b=100):
    g = x.copy()
    g[0] = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0] ** 2)
    g[1] = 2 * b * (x[1] - x[0] ** 2)
    return g.reshape((2, 1))


def grad(f, x, eps=1e-10):
    g = x.copy()
    for i in range(len(x)):
        move1 = x.copy()
        move1[i] += eps
        g[i] = (f(move1) - f(x)) / eps

    return g.reshape((2, 1))


def gradient_descend(f, x, eta=1e-3, tmax=1_000, eps=1e-10):
    X_history = np.zeros((tmax, 2))
    it = 0
    for i in range(tmax):
        g_n = grad(f, x)
        x_old = x.copy()
        x = x - eta * g_n
        X_history[i, :] = x.T
        it += 1

        delta = np.linalg.norm(x_old - x)
        if delta < eps:
            print("Stopped cycle at iteration {}".format(it))
            return x, X_history[: i + 1, :]

    return x, X_history


def visualize_GD(function, X_history, span=10.0):
    fig = plt.figure()
    ax = Axes3D(fig)

    # Define the dimensions of the x and y axis
    x = y = np.arange(-span, span, 0.05)

    # Create a rectangular grid with (x,y) coordinates for each point
    X, Y = np.meshgrid(x, y)

    # Calculte value of zs for each (x,y) pairing -- retuens a 1d array
    zs = np.array([function([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))])

    # Reshhape zs into a 'x' by 'y' matrix
    Z = zs.reshape(X.shape)

    # Calculate the coordinates of the Gradient Descent steps
    xg = X_history[:, 0]
    yg = X_history[:, 1]
    zg = zs = np.array([function([x, y]) for x, y in zip(np.ravel(xg), np.ravel(yg))])

    # Plot the function + the steps of the GD Algo
    ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap="jet", alpha=0.6)
    ax.plot(xg, yg, zg, color="g", marker="*", alpha=0.7)

    plt.show()


span = 3
x = np.random.uniform(-span, span, size=(2, 1))
x_new, X_history = gradient_descend(rosenbrock, x)
visualize_GD(rosenbrock, X_history, span=span)
