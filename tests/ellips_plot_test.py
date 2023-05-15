import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define ellipsoid parameters
a, b, c = 1, 2, 3  # Semi-axes lengths

A = np.array([[a, 0 ,0.1], [0, b, 0], [0, 0, c]])
# Create 3D ellipsoid coordinates
u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(-np.pi/2, np.pi/2, 100)
print(np.outer(np.cos(u), np.cos(v)).shape)
V = np.vstack([np.outer(np.cos(u), np.cos(v)).flatten(), np.outer(np.sin(u), np.cos(v)).flatten(), np.outer(np.ones_like(u), np.sin(v)).flatten()])

re = A.dot(V)

#x = a * np.outer(np.cos(u), np.cos(v)).flatten()
#y = b * np.outer(np.sin(u), np.cos(v)).flatten()
#z = c * np.outer(np.ones_like(u), np.sin(v)).flatten()

x = np.reshape(re[0,:], (100,100))
y = np.reshape(re[1,:], (100,100))
z = np.reshape(re[2,:], (100,100))

# Plot the ellipsoid
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()






