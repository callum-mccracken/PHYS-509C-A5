"""Q3 -- delta(chi2) statistic."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# In this problem you will use the delta(chi2) method
# to form a test statistic to discriminate between two types of events
# (let’s call them type A and type B).
# See Lecture 17, slide 20 for the definition of this test statistic.

# that definition:
# delta_chi2 = (x-mu_A)^T V_A^-1 (x-mu_A) - (x-mu_B)^T V_B^-1 (x-mu_B)

# You are given two large training sets of type A events and type B events.
# For each event two quantities have been measured:
# x (first column in the file) and y (second column),
# and there are 10,000 events in each training set.

# Use these training sets to define a delta(chi2) statistic
# analogous to what is shown on the slide referred to above.

# We'll need means and covariance matrices.

# first load the training datasets
train_x_A, train_y_A = np.loadtxt("trainingset_a.txt").T
train_x_B, train_y_B = np.loadtxt("trainingset_b.txt").T

# get means, standard deviations, and correlation coefficients
mu_x_train_A, sigma_x_train_A = np.mean(train_x_A), np.std(train_x_A)
mu_y_train_A, sigma_y_train_A = np.mean(train_y_A), np.std(train_y_A)
mu_x_train_B, sigma_x_train_B = np.mean(train_x_B), np.std(train_x_B)
mu_y_train_B, sigma_y_train_B = np.mean(train_y_B), np.std(train_y_B)
train_rho_A = np.corrcoef(train_x_A, train_y_A)[0,1]
train_rho_B = np.corrcoef(train_x_B, train_y_B)[0,1]

# construct the covariance matrices
V_A = np.array([
    [sigma_x_train_A**2, train_rho_A * sigma_x_train_A * sigma_y_train_A],
    [train_rho_A * sigma_x_train_A * sigma_y_train_A, sigma_y_train_A**2]])
V_B = np.array([
    [sigma_x_train_B**2, train_rho_B * sigma_x_train_B * sigma_y_train_B],
    [train_rho_B * sigma_x_train_B * sigma_y_train_B, sigma_y_train_B**2]])

# invert them
V_A_inv = np.linalg.inv(V_A)
V_B_inv = np.linalg.inv(V_B)

# now we have our delta(chi2) statistic
def delta_chi2(x, y):
    """
    The delta(chi2) statistic for our given training datasets.

    Uses the definition from the slides:
    delta(chi2) = (x-mu_A)^T V_A^-1 (x-mu_A) - (x-mu_B)^T V_B^-1 (x-mu_B)
    """
    vec_A = np.array([x, y]) - np.array([mu_x_train_A, mu_y_train_A])
    vec_B = np.array([x, y]) - np.array([mu_x_train_B, mu_y_train_B])
    part_A = np.dot(vec_A, np.matmul(V_A_inv, vec_A))
    part_B = np.dot(vec_B, np.matmul(V_B_inv, vec_B))
    return part_A - part_B

# You are also given two testing sets of type A and type B events.
# Apply your delta(chi2) statistic to each testing set
# and plot the distributions of the test statistic
# for each type of events on the same plot.

# load the test data
test_x_A, test_y_A = np.loadtxt("testingset_a.txt").T
test_x_B, test_y_B = np.loadtxt("testingset_b.txt").T

# evaluate delta(chi2) on the test data
test_dchi2_A = np.vectorize(delta_chi2)(test_x_A, test_y_A)
test_dchi2_B = np.vectorize(delta_chi2)(test_x_B, test_y_B)

# define bins so both histograms have the same widths / axis limits
bins=np.arange(-50, 50, 0.5)

# plot two histograms
plt.hist(test_dchi2_A, density=True, bins=bins, alpha=0.5, label="Test A")
plt.hist(test_dchi2_B, density=True, bins=bins, alpha=0.5, label="Test B")
plt.legend()
plt.title(r"$\Delta \chi^2$ Histograms For Test Datasets")
plt.xlabel(r"$\Delta \chi^2$")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("q3.png")

# For a given event with x=2.5, y=-0.5,
# what fraction of type A events have a higher value of this test statistic?
# What fraction of type B events have a higher value?
dchi2_25_05 = delta_chi2(2.5, -0.5)
n_A_higher = len([dchi2 for dchi2 in test_dchi2_A if dchi2 >= dchi2_25_05])
frac_A_higher = n_A_higher / len(test_dchi2_A)
n_B_higher = len([dchi2 for dchi2 in test_dchi2_B if dchi2 >= dchi2_25_05])
frac_B_higher = n_B_higher / len(test_dchi2_B)
print(f"{frac_A_higher=}, {frac_B_higher=}")
