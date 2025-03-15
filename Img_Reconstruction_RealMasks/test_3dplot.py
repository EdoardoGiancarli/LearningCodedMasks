import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
Z = np.random.normal(0, 30, (150, 150))
N, M = Z.shape  # Get the shape of the array

# Generate X and Y indices
X, Y = np.meshgrid(np.arange(M), np.arange(N))

# Flatten the arrays for bar3d input
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

# Define bar width
dx = dy = 1  # Size of each bar

# Create 3D figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Create the bar plot
ax.bar3d(X, Y, 0, dx, dy, Z, shade=True, cmap='viridis')

# Labels
ax.set_title('3D Bar Plot from 2D Array')

plt.show()