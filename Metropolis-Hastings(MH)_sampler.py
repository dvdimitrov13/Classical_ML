import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm


def sampler(
    data,
    samples=4,
    mu_init=0.5,
    proposal_width=0.5,
    plot=False,
    mu_prior_mu=0,
    mu_prior_sd=1.0,
):
    mu_current = mu_init
    posterior = [mu_current]
    for i in range(samples):
        # suggest new position
        mu_proposal = norm(mu_current, proposal_width).rvs()

        ### Introduced Log_likelihood ###

        # Compute likelihood by multiplying probabilities of each data point
        likelihood_current = np.sum(np.log(norm(mu_current, 1).pdf(data)))
        likelihood_proposal = np.sum(np.log(norm(mu_proposal, 1).pdf(data)))

        # Compute prior probability of current and proposed mu
        prior_current = np.log(norm(mu_prior_mu, mu_prior_sd).pdf(mu_current))
        prior_proposal = np.log(norm(mu_prior_mu, mu_prior_sd).pdf(mu_proposal))

        p_current = likelihood_current + prior_current
        p_proposal = likelihood_proposal + prior_proposal

        # Accept proposal? ## NEeds to be reworked if using log_likeli
        p_accept = np.exp(p_proposal - p_current)

        # Usually would include prior probability, which we neglect here for simplicity
        accept = np.random.rand() < p_accept

        if accept:
            # Update position
            mu_current = mu_proposal

        posterior.append(mu_current)

    return np.array(posterior)


data = np.random.randn(20)

np.random.seed(123)
posterior = sampler(data, samples=1000, mu_init=1.0)

plt.hist(posterior, density=True, bins=30)
plt.show()
