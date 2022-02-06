import numpy as np
import random
import matplotlib.pyplot as plt


def normalPDF(x, mu, sigma):
    num = np.exp(-1 / 2 * ((x - mu) / sigma) ** 2)
    den = np.sqrt(2 * np.pi) * sigma
    return num / den


def invGamma(x, a, b):
    non_zero = int(x >= 0)
    func = x ** (a - 1) * np.exp(-x / b)
    return non_zero * func


def lr_mcmc(X, y, hops=5_000):
    samples = []
    curr_a = random.gauss(1, 1)
    curr_b = random.gauss(2, 1)
    curr_s = random.uniform(3, 1)

    prior_a_curr = normalPDF(x=curr_a, mu=1, sigma=1)
    prior_b_curr = normalPDF(x=curr_b, mu=2, sigma=1)
    prior_s_curr = invGamma(x=curr_s, a=3, b=1)

    # Log likelihood of y
    log_lik_curr = sum(
        [
            np.log(normalPDF(x=y, mu=curr_b * x + curr_a, sigma=curr_s))
            for x, y in zip(X, y)
        ]
    )
    current_numerator = (
        log_lik_curr
        + np.log(prior_a_curr)
        + np.log(prior_b_curr)
        + np.log(prior_s_curr)
    )

    count = 0
    for i in range(hops):
        samples.append((curr_b, curr_a, curr_s))

        if count == 0:  # propose movement to b
            mov_a = curr_a
            mov_b = curr_b + random.uniform(-0.5, 0.5)
            mov_s = curr_s
            count += 1

        elif count == 1:  # propose movement to a
            mov_a = curr_a + random.uniform(-0.75, 0.75)
            mov_b = curr_b
            mov_s = curr_s
            count += 1

        else:  # propose movement to s
            mov_a = curr_a
            mov_b = curr_b
            mov_s = curr_s + random.uniform(-0.25, 0.25)
            count = 0

        prior_b_mov = normalPDF(x=mov_b, mu=2, sigma=1)
        prior_a_mov = normalPDF(x=mov_a, mu=1, sigma=1)
        prior_s_mov = invGamma(x=mov_s, a=3, b=1)
        if prior_s_mov <= 0:
            continue  # automatically reject because variance cannot equal 0.

        log_lik_mov = sum(
            [
                np.log(normalPDF(x=y, mu=mov_b * x + mov_a, sigma=mov_s))
                for x, y in zip(X, y)
            ]
        )
        movement_numerator = (
            log_lik_mov
            + np.log(prior_a_mov)
            + np.log(prior_b_mov)
            + np.log(prior_s_mov)
        )

        ratio = np.exp(movement_numerator - current_numerator)
        event = random.uniform(0, 1)
        if event <= ratio:
            curr_b = mov_b
            curr_a = mov_a
            curr_s = mov_s
            current_numerator = movement_numerator

    return samples


def gen_data(sample=100, intercept=10, beta=1.5, sigma=1):
    X = []
    Y = []
    for i in range(sample):
        x = np.random.uniform(0, 1000)
        y = intercept + x * beta + random.gauss(0, sigma)
        X.append(x)
        Y.append(y)

    return X, Y


X, y = gen_data(sample=1000)

test2 = lr_mcmc(X=X, y=y, hops=5_000)

b = [x[0] for x in test2]
a = [x[1] for x in test2]
s = [x[2] for x in test2]

plt.plot(np.arange(0, 4000, 1).tolist(), b[1000:])
plt.show()
