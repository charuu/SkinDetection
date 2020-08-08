import numpy as np
from scipy.stats import multivariate_normal


class MVND:
    # TODO: EXERCISE 2 - Implement mean and covariance matrix of given data
    def __init__(self, data: np.ndarray, c: float = 1.0):
        self.c = c  # Mixing coefficients. The sum of all mixing coefficients = 1.0.
        self.data = data
        self.mean = data.mean(axis=1)
        self.cov = np.cov(data)

    # TODO: EXERCISE 2 - Implement pdf and logpdf of a MVND
    def pdf(self, x: np.ndarray) -> float:
        return multivariate_normal.pdf(x, self.mean, self.cov)

    def logpdf(self, x: np.ndarray) -> float:
        return multivariate_normal.logpdf(x, self.mean, self.cov)