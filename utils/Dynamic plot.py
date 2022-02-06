from scipy.integrate import quad
import matplotlib.pyplot as plt
from IPython import display
import time


def f(x):
    return x


def Riemann_integral(f, n=100, a=0, b=1):
    h = (b - a) / n
    s = 0
    for i in range(n):
        s += f(a + i * h)
    F = h * s

    return F


anyl_ans, err = quad(f, 0, 1)
fig, ax = plt.subplots()

n = 100

x1 = [i for i in range(n)]
x2 = [0]
y1 = [anyl_ans for i in range(n)]
y2 = [0]

ax.set_xlim(0, n + 1)
ax.set_ylim(0, anyl_ans + 0.25 * anyl_ans)
ax.plot(x1, y1)

# Needs a jupyter notebook to work
for j in range(1, n):
    x2.append(j)
    y2.append(Riemann_integral(f, n=j))
    ax.scatter(x2, y2, color="red", marker="x", s=1.5)
    display.display(fig)
    display.clear_output(wait=True)
    time.sleep(0.05)
