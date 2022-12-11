"""Q2 -- Tukey's biweight."""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from tqdm import tqdm

# The file data.txt contains 100 pairs of x, y values
# you wish to fit to a straight line.
# The nominal uncertainty in the y measurements is 0.5,
# while the x values are known perfectly.
# However, there are clearly outliers in the data.
SIGMA_Y = 0.5

# number of bootstrap iterations for part B,
# better if larger, but takes longer
BOOTSTRAP_ITERATIONS = 10000

def tukey(x: float):
    """The "Tukey's Biweight" function."""
    if abs(x) >= 6:
        return 0
    return x * (1 - x**2/36)**2

def rho(x: float):
    """Intergral of tukey from -6 to x."""
    return quad(tukey, -6, x)  # equivalent to -inf to x

def linear_fit(x: np.ndarray, slope: float, intercept: float):
    """A line. Residual argument is for bootstrap."""
    return slope*x + intercept

def rho_sum(x: np.ndarray, y: np.ndarray, slope: float, intercept: float):
    """Return the thing we're supposed to minimize."""
    return np.sum(np.vectorize(rho)(
        (y - linear_fit(x, slope, intercept)) / SIGMA_Y))

def get_best_fit(x: np.ndarray, y: np.ndarray):
    """Get the best-fit slope and intercept, by minimizing rho_sum."""
    guess = [0.5, 5]
    def estimator(params):
        """Call rho_sum with params = (slope, intercept)."""
        return rho_sum(x, y, *params)
    return minimize(estimator, guess, method="Nelder-Mead").x

def main():
    """Do the question."""
    # Part A:
    # Fit the data to a straight line using the Tukey's biweight version of
    # an M-estimator, and report the slope and intercept that you get.
    x, y = np.loadtxt("data.txt").T
    slope_best, intercept_best = get_best_fit(x, y)
    print(f"best fit: {slope_best=}, {intercept_best=}")

    # Part B
    # Using those best-fit values,
    # calculate the residuals between the data and the fit.
    # Then use these residuals in the bootstrap method
    # to obtain uncertainty estimates for the fit you did in Part A.
    # For the bootstrap assume that the residuals are independent of x.

    # calculate the residuals
    residuals = y - linear_fit(x, slope_best, intercept_best)

    # estimate m, b for a bunch of iterations
    slope_estimates = np.empty(BOOTSTRAP_ITERATIONS)
    intercept_estimates = np.empty(BOOTSTRAP_ITERATIONS)
    for i in tqdm(range(BOOTSTRAP_ITERATIONS)):
        # pick x values and residuals independently
        x_i = np.random.choice(x, len(x), replace=True)
        r_i = np.random.choice(residuals, len(x), replace=True)
        # generate y dataset
        y_i = linear_fit(x_i, slope_best, intercept_best) - r_i
        # get best-fit slope, intercept
        slope_estimates[i], intercept_estimates[i] = get_best_fit(x_i, y_i)

    # the width on that new distribution gives the error on the estimator
    slope_error = np.std(slope_estimates)
    intercept_error = np.std(intercept_estimates)
    print(f"errors: {slope_error=} {intercept_error=}")

if __name__ == "__main__":
    main()
