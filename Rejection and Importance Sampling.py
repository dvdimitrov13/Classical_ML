import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import seaborn as sns


### Simple Monte Carlo example - generating exponential samples ###


def inverseFunction(lm, n):  # Inverse transform sampling
    u = stats.uniform.rvs(size=n)
    return -lm * np.log(1 - u)  # Inverse of the CDF


plt.hist(inverseFunction(0.1, 1000))
plt.show()


### Visualize the Accepante-Rejection ratio ###


def plot_halfnormal(n):
    x = np.linspace(0, 5, n)
    Y = []
    X = []
    for i in x:
        X.append(i)
        f = (math.sqrt(2 / math.pi)) * math.exp(i ** 2 / (-2))
        Y.append(f)
    return X, Y


def plot_exponential(n):
    x = np.linspace(0, 5, n)
    alpha_minus = math.sqrt((2 * math.exp(1)) / math.pi)
    Y = []
    X = []
    for i in x:
        X.append(i)
        f = math.exp(-i)
        Y.append(f * alpha_minus)
    return X, Y


def plot_ratio_halfnormexp(n):
    x = np.linspace(0, 5, n)
    Y = []
    X = []
    for i in x:
        X.append(i)
        f = math.exp(-0.5 * ((i - 1) ** 2))
        Y.append(f)
    return X, Y


X_hm, halfnormal = plot_halfnormal(100)
X_exp, exponential = plot_exponential(100)
X_r, ratio = plot_ratio_halfnormexp(100)

plt.figure()
plt.plot(X_hm, halfnormal, linewidth=2.0)
plt.plot(X_exp, exponential, linewidth=2.0)
plt.plot(X_r, ratio, linewidth=2.0)
plt.show()

### Acceptance-Rejection Sampling example ###

# Now we need to sample the positive half normal
# We will use the Rejection sampling technique


def rejection_sampling_hn(n=10000):
    x = []
    for i in range(n):
        support_prop = np.random.exponential()
        u = np.random.uniform()  # This is our coin flip
        AR_ratio = math.exp(-0.5 * ((support_prop - 1) ** 2))
        if u <= AR_ratio:
            x.append(support_prop)
    return x


# Turn halfnormal to normal
def FromHalfToStandard(l):
    new = []
    for el in l:
        v = np.random.uniform()
        if v <= 0.5:
            new.append(el)
        else:
            new.append(-el)
    return new


AR_sample_hn = rejection_sampling_hn()
fromhalftostandard = FromHalfToStandard(AR_sample_hn)
plt.figure()
plt.hist(AR_sample_hn, bins=50)
plt.figure()
plt.hist(fromhalftostandard, bins=50)
plt.show()


### Importance Sampling  example###


def f_x(x):
    return 1 / (1 + np.exp(-x))


def distribution(mu=0, sigma=1):
    # return probability given a value
    distribution = stats.norm(mu, sigma)
    return distribution


# pre-setting
n = 1000

mu_target = 3.5
sigma_target = 1
mu_appro = 3
sigma_appro = 1

p_x = distribution(mu_target, sigma_target)
q_x = distribution(mu_appro, sigma_appro)

plt.figure(figsize=[10, 4])

sns.distplot(
    [np.random.normal(mu_target, sigma_target) for _ in range(3000)],
    label="distribution $p(x)$",
)
sns.distplot(
    [np.random.normal(mu_appro, sigma_appro) for _ in range(3000)],
    label="distribution $q(x)$",
)

plt.title("Distributions", size=16)
plt.legend()
plt.show()

# value
s = 0
for i in range(n):
    # draw a sample
    x_i = np.random.normal(mu_target, sigma_target)
    s += f_x(x_i)
print("simulate value", s / n)

# calculate value sampling from a different distribution

value_list = []
for i in range(n):
    # sample from different distribution
    x_i = np.random.normal(mu_appro, sigma_appro)
    value = f_x(x_i) * (p_x.pdf(x_i) / q_x.pdf(x_i))

    value_list.append(value)

print("average {} variance {}".format(np.mean(value_list), np.var(value_list)))
