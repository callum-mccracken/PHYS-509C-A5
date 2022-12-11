"""Q1 -- t-Distribution."""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

X_OBS = 0.5  # the scale reading we've observed
CONF_LEVEL = 0.9  # we're looking for the 90% confidence interval

def likelihood(x, m):
    """
    Probability distribution of x, with mean m, std dev 1, and 3 dof.
    Derived in the text.
    """
    return 2 / np.pi / (1 + (x-m)**2)**2

def posterior(m, x=X_OBS):
    """Posterior, we must have m>=0 which is why 0 not -inf in the integral."""
    return likelihood(x, m) / quad(lambda n: likelihood(x, n), 0, np.inf)[0]

def integral_expression(upper_limit, x=X_OBS):
    """
    Return the integral expression we want to set to zero.
    (We want int(post.) = 0.9, so equivalently set int(post.) - 0.9 = 0)
    """
    # we want this to be zero so that the posterior integrates to CONF_LEVEL
    return quad(lambda m: posterior(m, x), 0, upper_limit)[0] - CONF_LEVEL

def ratio(x, m):
    """The likelihood ratio, for ordering purposes in the FC procedure."""
    best_m = x if x >= 0 else 0  # physical constraint: m >= 0
    return likelihood(x, m) / likelihood(x, best_m)

def main():
    """Do the question."""

    # The Bayesian part

    # Plot the integral expression so we can see roughly where it has a root
    upper_limits = np.linspace(0, 10, 1000)
    integrals = np.vectorize(integral_expression)(upper_limits)
    plt.plot(upper_limits, integrals)
    plt.savefig("q1_integrals.png")

    # this has a root around a=1, find it exactly
    bayesian_upper_limit = fsolve(integral_expression, [1])[0]
    print(f"{bayesian_upper_limit=}")

    # Feldman-Cousins

    # lower limit = 0 for m, find the upper one.

    # I tried to do this analytically, finding spots where r > thresh
    # that hurt my soul, so let's do integration here the easy numerical way
    # even if it feels less precise.
    d_x = 0.001
    x_limits = (-50, 50)  # the vast majority of the pdf is within here
    x = np.arange(*x_limits, d_x)

    # loop over m until we find one where the integral >= CONF_LEVEL
    d_m = 0.001
    for upper_limit_m in np.arange(0, 100, d_m):
        print(upper_limit_m)  # well possible upper limit at this point
        # we want to sort by decreasing r, and this is one way to have
        # r decrease naturally with m, it has a bit of a bump where
        # it increases but it seems to work eventually
        r_thresh = ratio(X_OBS, upper_limit_m)
        # essentially binning our data
        probs = likelihood(x, upper_limit_m) * d_x
        # calculate ratios for all x
        ratios = np.vectorize(ratio)(x, [upper_limit_m]*len(x))
        # integrate probs over the region where ratios > r_thresh
        integral = np.sum(probs[ratios >= r_thresh])
        # exit when CONF_LEVEL has been reached
        if integral >= CONF_LEVEL:
            # we've found the actual upper limit
            print(f"FC {upper_limit_m=}")
            break

if __name__ == "__main__":
    main()
