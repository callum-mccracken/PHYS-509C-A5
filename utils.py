import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform, poisson, norm, binom


def uniform_pdf(x, start, end):
    """
    Get a uniform probability density function.

    pdf(x) = 1/(end-start) if x in [start, end] else 0

    start, end = endpoints of distribution

    returns: the value of the pdf at x
    """
    return uniform(start, end-start).pdf(x)


def poisson_pmf(k, rate, time=1):
    """
    Get a Poisson probability mass function.

    pdf(k) = exp(-lambda)lambda^k/k!

    rate = mean expected events in some time
    (if you have a lambda value pass that as rate and leave time=1)

    time = that time

    returns: the value of the pmf at k
    """
    lamb = rate * time  # lambda
    return poisson(lamb).pmf(k)


def gaussian_pdf(x, mu, sigma):
    """
    Get a Gaussian probability density function.

    pdf(x) = 1/(sqrt(2pi)sigma)exp(-(x-mu)^2/(2sigma^2))

    x = value to evaluate pdf
    mu = mean
    sigma = standard deviation

    returns: the value of the pdf at x
    """
    return norm(mu, sigma).pdf(x)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """
    Get a binomial probability mass function.

    pmf(k) = nCk p^k (1-p)^{n-k}

    k = value to evaluate pmf
    n = population size
    p = probability of success

    returns: the value of the pmf at k
    """
    return binom(n, p).pmf(k)


def one_param(f):
    """
    Often we have functions with many parameters we want to minimize.
    However, the minimizer requires functions with only one parameter.
    Use this to get the same function back as a one-parameter thing.
    """
    def new_f(parameters):
        return f(*parameters)
    return new_f



def main():
    """make some plots"""
    unif_x = np.linspace(0, 4, 100)
    plt.title("uniform")
    plt.plot(unif_x, uniform_pdf(unif_x, start=1, end=2))
    plt.show()

    poisson_k = np.array(range(20))
    plt.title("poisson")
    plt.bar(poisson_k, poisson_pmf(poisson_k, rate=5))
    plt.show()

    gauss_x = np.linspace(-3, 3, 1000)
    plt.title("gaussian")
    plt.bar(gauss_x, gaussian_pdf(gauss_x, mu=0, sigma=1))
    plt.show()


if __name__ == "__main__":
    main()
