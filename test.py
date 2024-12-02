import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Define the target function (example: a sinusoidal function with two variables)
def target_function(x, y):
    return np.sin(x) * np.cos(y)

# Simulated Gaussian Process predicted mean and variance
def gaussian_process_mean(x, y):
    return np.sin(x) + np.cos(y) - 0.5

def gaussian_process_variance(x, y):
    return 0.1 + 0.4 * np.exp(-((x - 3)**2 + (y - 3)**2) / 2)

# Simulated acquisition function
def acquisition_function(x, y):
    return gaussian_process_mean(x, y) + gaussian_process_variance(x, y)

# Generate a grid
x = np.linspace(0, 6, 100)
y = np.linspace(0, 6, 100)
X, Y = np.meshgrid(x, y)

# Compute values for the plots
Z_target = target_function(X, Y)
Z_mean = gaussian_process_mean(X, Y)
Z_variance = gaussian_process_variance(X, Y)
Z_acquisition = acquisition_function(X, Y)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(8, 6 ))
fig.suptitle("Bayesian Optimization in Action", fontsize=16)

print(X)
print(X.shape)
print(Z_mean)
print(Z_mean.shape)
# Plot Gaussian Process Predicted Mean
im1 = axes[0, 0].contourf(X, Y, Z_mean, cmap=cm.viridis)
axes[0, 0].scatter([2, 4, 3], [2, 3, 4], color="black")  # Example data points
axes[0, 0].set_title("Gaussian Process Predicted Mean")
fig.colorbar(im1, ax=axes[0, 0])

# Plot Target Function
im2 = axes[0, 1].contourf(X, Y, Z_target, cmap=cm.viridis)
axes[0, 1].scatter([2, 4, 3], [2, 3, 4], color="black")  # Example data points
axes[0, 1].set_title("Target Function")
fig.colorbar(im2, ax=axes[0, 1])

# Plot Gaussian Process Variance
im3 = axes[1, 0].contourf(X, Y, Z_variance, cmap=cm.viridis)
axes[1, 0].set_title("Gaussian Process Variance")
fig.colorbar(im3, ax=axes[1, 0])

# Plot Acquisition Function
im4 = axes[1, 1].contourf(X, Y, Z_acquisition, cmap=cm.viridis)
axes[1, 1].set_title("Acquisition Function")
fig.colorbar(im4, ax=axes[1, 1])

# Show the plot
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()