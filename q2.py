"""
Q2 -- Tukey's biweight

The file data.txt contains 100 pairs of x, y values
you wish to fit to a straight line.
The nominal uncertainty in the y measurements is 0.5,
while the x values are known perfectly.
However, there are clearly outliers in the data.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from tqdm import tqdm

# uncertainty on each point is constant.
SIGMA = 0.5
# number of bootstrap iterations, better if larger, but takes longer
BOOTSTRAP_ITERATIONS = 10000

def tukey(x: float):
    """The "Tukey's Biweight" function."""
    if abs(x) >= 6:
        return 0
    return x * (1 - x**2/36)**2

def rho(x: float):
    """Intergral of tukey from -6 to x."""
    return quad(tukey, -6, x)  # equivalent to -inf to x

def linear_fit(x: np.ndarray, m: float, b: float):
    """A line. Residual argument is for bootstrap."""
    return m*x + b

def rho_sum(x: np.ndarray, y: np.ndarray, m: float, b: float):
    """Return the thing we're supposed to minimize."""
    z = (y - linear_fit(x, m, b)) / SIGMA
    return np.sum(np.vectorize(rho)(z))

def get_best_fit_m_b(x: np.ndarray, y: np.ndarray):
    """Get the best-fit m and b, by minimizing rho_sum."""
    guess = [0.5, 5]
    def estimator(params):
        """Call rho_sum with params = (m, b)."""
        return rho_sum(x, y, *params)
    return minimize(estimator, guess, method="Nelder-Mead").x

def main():
    """Do the question."""
    # Part A:
    # Fit the data to a straight line using the Tukey's biweight version of
    # an M-estimator, and report the slope and intercept that you get.
    x, y = np.loadtxt("data.txt").T
    m_best, b_best = get_best_fit_m_b(x, y)
    print(f"best fit: {m_best=}, {b_best=}")

    # Part B
    # Using those best-fit values,
    # calculate the residuals between the data and the fit.
    # Then use these residuals in the bootstrap method
    # to obtain uncertainty estimates for the fit you did in Part A.
    # For the bootstrap assume that the residuals are independent of x.

    # calculate the residuals
    residuals = y - linear_fit(x, m_best, b_best)

    # estimate m, b for a bunch of iterations
    m_estimates = np.empty(BOOTSTRAP_ITERATIONS)
    b_estimates = np.empty(BOOTSTRAP_ITERATIONS)
    for i in tqdm(range(BOOTSTRAP_ITERATIONS)):
        # pick x values and residuals independently
        x_i = np.random.choice(x, len(x), replace=True)
        r_i = np.random.choice(residuals, len(x), replace=True)
        # generate y dataset
        y_i = linear_fit(x_i, m_best, b_best) - r_i
        # get best-fit m, b
        m_estimates[i], b_estimates[i] = get_best_fit_m_b(x_i, y_i)

    # the width on that new distribution gives the error on the estimator
    b0_error = np.std(b_estimates)
    m0_error = np.std(m_estimates)
    print(f"errors: {m0_error=} {b0_error=}")

if __name__ == "__main__":
    main()
