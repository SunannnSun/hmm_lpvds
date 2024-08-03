import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Sample data
data = np.random.rand(50, 2)  # 50 points in 2D space

# Set up the figure and axis
fig, ax = plt.subplots()
sc = ax.scatter([], [])
    
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Define update function for animation
def update(frame):
    ax.clear()
    start = max(0, frame - 4)
    end = frame + 1
    current_points = data[start:end]

    # Calculate opacities
    alphas = np.linspace(0.2, 1, current_points.shape[0])
    
    # Plot points with varying opacities
    for i, point in enumerate(current_points):
        ax.scatter(point[0], point[1], color='blue', alpha=alphas[i])

    return sc,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(data), interval=200, blit=False)

# Save or show animation
plt.show()