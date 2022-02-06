###
# Applying Newton-Rhapson to Logistic regression to estimate parameters (IRLS)
###

import numpy as np

# Simulated data
np.random.seed(1337)
p = 5  # number of variables
n = 1000  # number of observations
X = np.ones(n, dtype=int)  # intercept
X = np.reshape(X, (n, 1))
X = np.hstack((X, np.random.normal(size=(n, p))))  # gaussian covariates
true_beta = np.random.randint(p + 1, size=p + 1)  # true coefficients
prob = np.exp(np.dot(X, true_beta)) / (1 + np.exp(np.dot(X, true_beta)))
Y = np.random.binomial(1, prob).reshape((n, 1))  # simulated dependent variable

print("True beta:", true_beta)


def logit(X, Y, epsilon):
    # X = covariates
    # Y = dependent variable
    # epsilon = threshold for convergence
    n, p = np.shape(X)
    # Initial values of the algorithms
    b_0 = np.zeros((p, 1))
    probs = np.exp(np.dot(X, b_0)) / (1 + np.exp(np.dot(X, b_0)))
    W = np.diag(
        (probs * (1 - probs)).reshape(
            n,
        )
    )  # X^T*W*X equals the Hessian
    Z = np.dot(X, b_0) + np.dot(np.linalg.inv(W), Y - probs)
    # boolean
    convergence = False
    while not convergence:
        # new estimate
        b = np.linalg.multi_dot(
            [np.linalg.inv(np.linalg.multi_dot([X.T, W, X])), X.T, W, Z]
        )
        probs = np.exp(np.dot(X, b)) / (1 + np.exp(np.dot(X, b)))
        W = np.diag(
            (probs * (1 - probs)).reshape(
                n,
            )
        )
        Z = np.dot(X, b) + np.dot(np.linalg.inv(W), Y - probs)
        if np.linalg.norm(b - b_0) / (np.linalg.norm(b_0) + epsilon) < epsilon:
            # relative convergence criterion
            convergence = True
            print(
                "\nConvergence reached with value:",
                np.linalg.norm(b - b_0) / (np.linalg.norm(b_0) + epsilon),
            )
        b_0 = b  # old value
    return b


# Run the algorithm
beta = logit(X, Y, epsilon=0.000001)
print("Beta:", beta.reshape((p + 1,)))  # estimated beta
print("True beta:", true_beta)  # true beta

# using built-in function of Python
from sklearn.linear_model import LogisticRegression

fit = LogisticRegression(fit_intercept=False).fit(
    X, Y.reshape((n,))
)  # we do not fit the intercept since we added a column in X with 1 at each position
print("\nEstimate:", fit.coef_)
print("True beta:", true_beta)
