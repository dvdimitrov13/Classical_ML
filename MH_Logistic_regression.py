import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.discrete.discrete_model import Probit

np.random.seed(1337)


def gen_data(n, d):
    # n the number of observations
    # d the number of parameters

    X = np.ones(n, dtype=int)  # intercept
    X = np.reshape(X, (n, 1))
    X = np.hstack((X, np.random.normal(size=(n, d - 1))))  # gaussian covariates

    # Initialize true params
    true_theta = np.random.normal(loc=0.25, scale=0.5, size=d).T  # true coefficients

    # By making use of the probit link, compute the probabilities p
    p = stats.norm.cdf(X.dot(true_theta))

    # Simulate y using the probalities and a binomial distribution
    y = np.random.binomial(1, p, size=None)

    return y, X, true_theta


def MH_logistic_reg(y, X, iter, burn_in, informative=True):
    n, d = tuple(X.shape)

    # initialize theta
    theta = np.zeros(d).T

    # Create a theta storage array
    theta_chain = np.zeros((iter, d))

    # In a previous simplified version of MH we sampled proposals
    # for each parameter now we are going to sample all params
    # from a multivariate normal with variance equal to the
    # inverse of the Fisher Information matrix

    # Probit is used instead of logit for ease of implemetation
    V = np.linalg.inv(-Probit(y, X).hessian(theta))

    for i in tqdm(range(iter)):

        proposal_theta = np.random.multivariate_normal(theta, V)

        prior = lambda theta: 1

        # Using an informative prior
        if informative:
            prior = lambda theta: stats.multivariate_normal.pdf(
                theta, mean=theta, cov=(np.eye(d) * 10)
            )

        # A log likelihood can be implemented to avoid underflow
        likelihood = lambda theta: np.exp(Probit(y, X).loglike(theta))

        acceptance = min(
            1,
            (prior(proposal_theta) * likelihood(proposal_theta))
            / (prior(theta) * likelihood(theta)),
        )

        u = np.random.uniform()
        if u < acceptance:
            theta = proposal_theta
        theta_chain[i, :] = theta.reshape((1, d))

    theta_chain = theta_chain[
        burn_in:,
    ]

    return theta_chain


# Run algorithm
n, d = 400, 4
iter, burn_in = 10000, 2000

y, X, true_theta = gen_data(n, d)
theta_chain = MH_logistic_reg(y, X, iter, burn_in)
mean_theta_est = theta_chain.mean(axis=0)


# Visualize results
fig, ax = plt.subplots(2, d, figsize=(24, 12))
for i in range(2):
    if i == 0:
        for j in range(d):
            ax[i][j].hist(theta_chain[:, j], bins=40)
            ax[i][j].axvline(
                x=mean_theta_est[j],
                ymin=0,
                ymax=400,
                label="Final value of Theta",
                color="tomato",
            )
            ax[i][j].axvline(
                x=true_theta[j],
                ymin=0,
                ymax=400,
                label="True value of Theta",
                color="plum",
            )
            ax[i][j].set_title("Distribution of parameter " + str(j) + " post burn-in")
    elif i == 1:
        for j in range(d):
            ax[i][j].plot(theta_chain[:, j], zorder=5, linewidth=0.5)
            ax[i][j].hlines(
                y=mean_theta_est[j],
                xmin=0,
                xmax=iter - burn_in,
                label="Final value of Theta",
                zorder=10,
                color="tomato",
            )
            ax[i][j].hlines(
                y=true_theta[j],
                xmin=0,
                xmax=iter - burn_in,
                label="True value of Theta",
                zorder=10,
                color="plum",
            )
            ax[i][j].set_title("Convergence of parameter " + str(j))

print(true_theta)
print(mean_theta_est)
plt.show()
