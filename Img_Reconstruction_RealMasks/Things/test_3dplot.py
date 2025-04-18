import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example data
N, M = 150, 150

# Generate X and Y indices
X, Y = np.meshgrid(np.arange(-M//2, M//2), np.arange(-N//2, N//2))
Z = np.exp(-(X**2 + Y**2)/20)

# Flatten the arrays for bar3d input
X = X.flatten()
Y = Y.flatten()
Z = Z.flatten()

# Define bar width
dx = dy = 1  # Size of each bar

# Create 3D figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Create the bar plot
ax.bar3d(X, Y, 0, dx, dy, Z, shade=True, cmap='viridis')
plt.tight_layout()

# Labels
ax.set_title('3D Bar Plot from 2D Array')

plt.show()