import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulated data
np.random.seed(1337)
n = 400  # number of observations
c = 3  # number of components
true_p = np.random.dirichlet(np.ones(c))  # true weights
true_mu = np.random.randint(size=c, low=-20, high=20)  # true means
true_sigma2 = np.random.randint(size=c, low=1, high=5)  # true variances

print("True means:", true_mu)
print("True variances:", true_sigma2)
print("True weights:", true_p)


def data_gen(n, p, mu, sigma):
    # generate n observations from a mixture of normal distributions
    c = len(p)
    cum_p = np.cumsum(p)
    samples = np.zeros((n, c))
    output = np.zeros(n)
    # sample n observations from each component
    for j in range(c):
        samples[:, j] = np.random.normal(loc=mu[j], scale=sigma[j], size=n)
    # sample the component from which each observations is sampled
    un = np.random.uniform(size=n)
    for i in range(n):
        j = np.where(cum_p >= un[i])[0][0]
        output[i] = samples[i, j]
    return output


Y = data_gen(n, true_p, true_mu, np.sqrt(true_sigma2))
plt.hist(Y, bins=50, density=True, edgecolor="black")
plt.title("Histogram of observed data")
plt.show()

# EM algorithm
def EM(Y, c, epsilon):
    # initial values
    p0 = np.ones(c) / c
    mu0 = Y[:c]
    sigma20 = np.ones(c) * np.var(Y)

    # store the values
    mu = np.zeros(c)
    sigma2 = np.zeros(c)
    p = np.zeros(c)

    convergence = False
    lik = []  # likelihood
    while not convergence:
        # expectation step
        Gamma = np.zeros((n, c))
        aux = 0
        # sample weights
        for i in range(n):
            # to compute log-likelihood
            aux = aux + np.log(
                np.sum(p0 * norm.pdf(Y[i], loc=mu0, scale=np.sqrt(sigma20)))
            )
            Gamma[i, :] = p0 * norm.pdf(Y[i], loc=mu0, scale=np.sqrt(sigma20))
            Gamma[i, :] = Gamma[i, :] / sum(Gamma[i, :])
        lik.append(aux)  # store log-likelihood

        # maximization step
        # computed weighted estimators
        for j in range(c):
            tot = sum(Gamma[:, j])
            mu[j] = sum(Gamma[:, j] * Y) / tot
            sigma2[j] = sum(Gamma[:, j] * (Y - np.ones(n) * mu[j]) ** 2) / tot
            p[j] = tot / n
        if (
            np.linalg.norm(mu + sigma2 - mu0 - sigma20)
            / (np.linalg.norm(mu0 + sigma20) + epsilon)
            < epsilon
        ):
            # relative convergence criterion
            convergence = True
            print(
                "\nConvergence reached with value:",
                np.linalg.norm(mu + sigma2 - mu0 - sigma20)
                / (np.linalg.norm(mu0 + sigma20) + epsilon),
            )
        # assign old values to p0
        p0 = np.copy(
            p
        )  # we need to copy, the equality would just assign the same pointer
        mu0 = np.copy(mu)
        sigma20 = np.copy(sigma2)
    return (p, mu, sigma2, lik)


p, mu, sigma2, lik = EM(Y, c, epsilon=0.0000001)
print("Means:", mu)
print("Variances:", sigma2)
print("Weights:", p)

plt.plot(lik)  # plot of the log-likelihood
plt.title("Log-likelihood")
# plt.savefig("lik.pdf")
plt.show()
