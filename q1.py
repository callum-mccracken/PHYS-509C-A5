"""Q1 -- t-Distribution"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

X_OBS = 0.5
CONF_LEVEL = 0.9

def likelihood(x, m):
    """Probability distribution of x, with mean m, std dev 1, and 3 dof."""
    return 2 / np.pi / (1 + (x-m)**2)**2

def posterior(m, x=X_OBS):
    """Posterior, note that we must have m>=0 which is why 0 not -inf."""
    return likelihood(x, m) / quad(lambda n: likelihood(x, n), 0, np.inf)[0]

def integral_expression(upper_limit, x=X_OBS):
    """Return the integral expression we want to set to zero."""
    # we want this to be zero so that the posterior integrates to CONF_LEVEL
    return quad(lambda m: posterior(m, x), 0, upper_limit)[0] - CONF_LEVEL

def ratio(x, m):
    """Felfman-Cousins likelihood ratio for ordering."""
    best_m = x if x >= 0 else 0  # physical constraint: m >= 0
    return likelihood(x, m) / likelihood(x, best_m)

def main():
    """Do the question."""

    # Bayesian

    # plot the integral expression so we can see roughly where it has a root
    upper_limits = np.linspace(0, 10, 1000)
    integrals = np.vectorize(integral_expression)(upper_limits)
    plt.plot(upper_limits, integrals)
    plt.savefig("q1_integrals.png")

    # this has a root around a=1, find it exactly
    bayesian_upper_limit = fsolve(integral_expression, [1])[0]
    print(f"{bayesian_upper_limit=}")

    # Feldman-Cousins

    # lower limit = 0 for physical reasons, find the upper one

    # I tried to do this analytically, finding spots where r > thresh
    # that hurt my soul, so let's do integration here the easy numerical way
    dx = 0.001
    x_limits = (-50, 50)  # the vast majority of the pdf is within here
    x = np.arange(*x_limits, dx)

    # loop over m until we find one where the integral over probs
    dm = 0.001
    for upper_limit_m in np.arange(0, 100, dm):
        print(upper_limit_m)  # well possible upper limit at this point
        # we want to sort by decreasing r, and this is one way to have
        # r decrease naturally with m, it has a bit of a bump where
        # it increases but it seems to work eventually
        r_thresh = ratio(X_OBS, upper_limit_m)
        # essentially binning our data
        probs = likelihood(x, upper_limit_m) * dx
        # calculate ratios for all x
        ratios = np.vectorize(ratio)(x, [upper_limit_m]*len(x))
        # "integrate" probs over where ratios > r_thresh
        integral = np.sum(probs[ratios >= r_thresh])
        # exit when needed
        if integral >= CONF_LEVEL:
            print(f"FC {upper_limit_m=}")
            break
        if upper_limit_m == 100:
            raise ValueError("Something broke!")

if __name__ == "__main__":
    main()
