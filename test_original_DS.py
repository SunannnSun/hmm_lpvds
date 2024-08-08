import numpy as np
# from src.util import optimize_tools, load_tools, process_tools
from src.util import optimize_tools
from src.lpvds.src.lpvds_class import lpvds_class
from src.lpvds.src.dsopt.dsopt_class import dsopt_class

from src.lpvds.src.util import load_tools, plot_tools
from matplotlib.collections import LineCollection
import matplotlib

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import time



input_message = '''
Please choose a data input option:
1. PC-GMM benchmark data
2. LASA benchmark data
3. Damm demo data
4. DEMO
Enter the corresponding option number: '''
# input_opt  = input(input_message)
input_opt = 2

x, x_dot, x_att, x_init = load_tools.load_data(int(input_opt))


lpvds = lpvds_class(x, x_dot, x_att)
lpvds.begin()


# evaluate results
x_test_list = []
for x_0 in x_init:
    x_test_list.append(lpvds.sim(x[0, :].reshape(1, -1), dt=0.01))







x_test = x_test_list[0]


fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(x[:, 0], x[:, 1], color='k', s=5, alpha=0.4, label='original data')

# Set up the plot limits
# ax.set_xlim(-50, 10)
# ax.set_ylim(-10, 20)

# Generate data
xx = np.array(x_test)[:, 0]
yy = np.array(x_test)[:, 1]


colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']

sequence_arr_test = np.zeros((xx.shape[0], ), dtype=int)
labels = np.take(colors, sequence_arr_test)

# Create line segments
points = np.array([xx, yy]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Initialize LineCollection
lc = LineCollection(segments, linewidth=8)
ax.add_collection(lc)



streamline_list = []
# for s in range(len(pos_ds)):


x_min, x_max = ax.get_xlim()
y_min, y_max = ax.get_ylim()

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plot_sample = 50
x_mesh,y_mesh = np.meshgrid(np.linspace(x_min,x_max,plot_sample),np.linspace(y_min,y_max,plot_sample))
X = np.vstack([x_mesh.ravel(), y_mesh.ravel()])


ds = lpvds
A = ds.A
att = ds.x_att
gamma = ds.damm.logProb(X.T)
for k in np.arange(len(A)):
    if k == 0:
        dx = gamma[k].reshape(1, -1) * (A[k] @ (X - att.T))  # gamma[k].reshape(1, -1): [1, num] dim x num
    else:
        dx +=  gamma[k].reshape(1, -1) * (A[k] @ (X - att.T)) 
u1 = dx[0,:].reshape((plot_sample,plot_sample))
v1 = dx[1,:].reshape((plot_sample,plot_sample))



ax.streamplot(x_mesh,y_mesh,u1,v1, density=2.0, color="black", arrowsize=1.1, arrowstyle="->")
scatter1 = ax.scatter(x_att[0, 0], x_att[0, 1], s=150, facecolors='none', edgecolors='magenta', linewidths=4)

ax.set_title(f"Original DS; single task sequence")





    

def init():
    """Initialize the background of the plot."""
    lc.set_segments([])

    return lc,

def update(frame):
    """Update the plot with new data for each frame."""

    # Update segments up to the current frame   
    start = max(0, frame - 10)
    end = frame + 1
    current_segments = segments[start:end]
    # current_segments = segments[:frame]

    # Set the colors based on the labels
    colors = [labels[i] for i in np.arange(start, end)]
    # colors = [labels[sequence_arr_test[frame]]] * current_segments.shape[0]
    lc.set_segments(current_segments)
    lc.set_color(colors)



    return lc, 

# Create the animation
ani = FuncAnimation(fig, update, interval=10, frames=len(x), init_func=init, blit=False, repeat=False)

ani.save("animation_DS.gif", writer=PillowWriter(fps=45))

import matplotlib.animation as animation
FFwriter = animation.FFMpegWriter(fps=45)
ani.save('animation_DS.mp4', writer = FFwriter)

# Show the animation
plt.show()
