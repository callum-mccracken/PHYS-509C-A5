from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from q1 import likelihood as P
from tqdm import tqdm

X_OBS = 0.5
BIN_RESOLUTION = 1# 0.001

def likelihood(x, alpha):
    """General likelihood function."""
    return quad(lambda y: P(y, m=alpha), x-0.5*BIN_RESOLUTION, x+0.5*BIN_RESOLUTION)[0]

def fc_ratio(x, alpha):
    """General Feldman-Cousins ratio."""
    # highest likelihood for this distribution is at the observed value
    x_best = X_OBS
    return likelihood(x, alpha) / likelihood(x_best, alpha)

class ratio_data_object():
    def __init__(self, x, alpha) -> None:
        """
        Create an instance (calculate likelihood and ratio,
        store values at which they were calculated.
        """
        self.x = x
        self.alpha = alpha
        self.prob = likelihood(x, alpha)
        self.ratio = fc_ratio(x, alpha)
    def __lt__(self, other):
        """Define how to sort this kind of object so we can call sorted()."""
        return self.ratio > other.ratio

def main():
    """Print FC intervals."""
    # m=alpha can be any positive number, for our purposes here, truncate
    min_alpha = 0
    max_alpha = 10

    # x here can range from -infty to +infty.
    # But for the purposes here use finite values
    xs = np.arange(min_alpha-10, max_alpha+10, BIN_RESOLUTION)

    # show the posterior distribution for one particular m=alpha value
    alpha = 2
    pdf = [likelihood(x_i, alpha) for x_i in xs]
    ratio = [fc_ratio(x_i, alpha) for x_i in xs]
    plt.plot(xs, pdf, label="pdf")
    plt.plot(xs, ratio, label="R")
    plt.legend()
    plt.show()

    print(sum(pdf))

    # for each x value we'll come up with lower and upper limits
    upperlimit = {x_i: -np.inf for x_i in xs}  # start low to set limits later
    lowerlimit = {x_i: np.inf for x_i in xs}  # start high

    # for a given true value of alpha
    d_alpha = (max_alpha - min_alpha)/1000
    for alpha in tqdm(np.arange(min_alpha, max_alpha, d_alpha)):
        # calculate ratios
        ratio_objs = [ratio_data_object(x_i, alpha) for x_i in xs]

        # add likelihoods together,
        # in order of highest-to-lowest ratio,
        # until you have something larger than 90%
        ratio_objs = sorted(ratio_objs)
        # start with the highest-ratio one
        index = 0

        # find min, max x boundaries for this alpha
        # (such that sum of probs within lower and upper = 0.9)
        # NOTE: here we assume the interval is contiguous, doesn't have to be!
        lower_bound = max(xs) + 1  # start high, will set this later
        upper_bound = min(xs) - 1  # start low, will set this later
        totalprob=0
        while totalprob < 0.9:
            if index >= len(ratio_objs):
                raise ValueError(totalprob)
            prob = ratio_objs[index].prob
            x = ratio_objs[index].x
            totalprob += prob
            if x < lower_bound:
                lower_bound = x
            if x > upper_bound:
                upper_bound = x
            index += 1

        # update upper/lower limits at the bounds we found
        if alpha > upperlimit[lower_bound]:
            upperlimit[lower_bound]=alpha
        if alpha < lowerlimit[upper_bound]:
            lowerlimit[upper_bound]=alpha

    for x_i in xs:
        print(x_i, lowerlimit[x_i], upperlimit[x_i])


if __name__ == "__main__":
    main()
