import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

# simulate data from a standard gaussian random variable
np.random.seed(1337)
n = 1000  # number of observations
y = np.random.normal(size=n, loc=0)
median_y = np.quantile(y, q=0.5)  # empirical median
mean_y = np.mean(y)  # empirical mean
print("Estimate of the median:", median_y)
print("Estimate of the mean:", mean_y)

# We want to use the bootstrap to approximate the distribution of the
# empirical median and the empirical mean
B = 1000  # number of bootstrap samples
boot_median = np.zeros(B)
boot_mean = np.zeros(B)
for j in range(B):
    boot_y = resample(y, replace=True)  # resample with replacement
    boot_median[j] = np.quantile(boot_y, q=0.5)
    boot_mean[j] = np.mean(boot_y)

# histograms for boostrapped median and mean
plt.hist(boot_median, bins=50, facecolor="g", alpha=0.75, edgecolor="black")
plt.title("Histogram of bootstrapped medians")
plt.show()
plt.hist(
    boot_mean, bins=50, range=(-0.1, 0.1), facecolor="r", alpha=0.75, edgecolor="black"
)
plt.title("Histogram of bootstrapped means")
plt.show()

# we can compare the bootstrapped 95%-confidence interval for the mean with the exact one
print(
    "Bootstrapped confidence interval for the mean:\n",
    np.quantile(boot_mean, q=0.025),
    np.quantile(boot_mean, q=0.975),
)
print(
    "True confidence interval for the mean:\n",
    mean_y - 1.96 / np.sqrt(n),
    mean_y + 1.96 / np.sqrt(n),
)
