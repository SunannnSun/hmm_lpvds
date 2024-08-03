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


N1 = 420
N_att = 0

sequence_arr = np.zeros((x.shape[0], ), dtype=int)
sequence_arr[N1: ] = 1

T_start = time.time()
x_1 = x[sequence_arr==0, :]
x_dot_1 = x_dot[sequence_arr==0, :]

x_2 = x[sequence_arr==1, :]
x_dot_2 = x_dot[sequence_arr==1, :]

N_1 = x_1.shape[0]
x_att_1 = x[N_1+N_att, :].reshape(1, -1)
x_att_2 = x_att

pos_ds_1 = lpvds_class(x_1, x_dot_1, x_att_1)
pos_ds_2 = lpvds_class(x_2, x_dot_2, x_att_2)

pos_ds_1.begin()
pos_ds_2.begin()

pos_ds = [pos_ds_1, pos_ds_2]
x_att  = [x_att_1, x_att_2]

'''Simulate'''

x_init = x_init[0].reshape(1, -1)

N_C = 10
confidence_arr = np.zeros((x.shape[0], ))
c_i = 10

n_S = 2
alpha = np.zeros((n_S, ))

dt = 0.01

thld = 0.8


# initialize first state
x_test = [x_init[0].reshape(1, -1)]
x_dot_test = [x_dot[0, :].reshape(1, -1)]  # actual velocity
x_dot_pred = [x_dot[0, :].reshape(1, -1)]  # predicted velocity

sequence_arr_test = [sequence_arr[0]] 
confidence_arr_test = [10]


gmm_prob_list = []
# for i in range(1500):
i = 0
while np.linalg.norm(x_test[-1] -x_att_2) > 0.1:
    i+=1
    # Step 1: Compute transition probability matrix using the previous state
    x_prev = x_test[-1] 
    s_prev = sequence_arr_test[-1]

    #P(b_i=1|s_i=s1)
    p_b_1_s1 = 1 if np.linalg.norm(x_prev - x_att[0][0, :]) < thld else 0
    #P(b_i=0|s_i=s1)
    p_b_0_s1 = 1 - p_b_1_s1

    #P(b_i=1|s_i=s2)
    p_b_1_s2 = 1 if np.linalg.norm(x_prev - x_att[1][0, :]) < thld else 0
    #P(b_i=0|s_i=s2)
    p_b_0_s2 = 1 - p_b_1_s2
    
    #P(s_ip1=s2|s_i=s1, b_i=1)
    p_s_ip1_s2_s_i_s1_b_1 = 1
    #P(s_ip1=s2|s_i=s1, b_i=0)
    p_s_ip1_s2_s_i_s1_b_0 = 0

    #P(s_ip1=s1|s_i=s1, b_i=1) 
    p_s_ip1_s1_s_i_s1_b_1 = 0
    #P(s_ip1=s1|s_i=s1, b_i=0)
    p_s_ip1_s1_s_i_s1_b_0 = 1

    #P(s_ip1=s2|s_i=s2, b_i=1) 
    p_s_ip1_s2_s_i_s2_b_1 = 1 # special case because s2 is the last state, so it always remains regardless
    #P(s_ip1=s2|s_i=s2, b_i=0)
    p_s_ip1_s2_s_i_s2_b_0 = 1

    """Integrate over all possible b_i given s_i"""
    #P(s_ip1=s1|s_i=s1) = P(s_ip1=s1|s_i=s1, b_i=0) * P(b_i=0|s_i=s1) + P(s_ip1=s1|s_i=s1, b_i=1) * P(b_i=1|s_i=s1)
    P_s_ip1_s1_s_i_s1 =  p_s_ip1_s1_s_i_s1_b_0 * p_b_0_s1 + p_s_ip1_s1_s_i_s1_b_1 * p_b_1_s1

    #P(s_ip1=s2|s_i=s1) 
    P_s_ip1_s2_s_i_s1 =  p_s_ip1_s2_s_i_s1_b_0 * p_b_0_s1 + p_s_ip1_s2_s_i_s1_b_1 * p_b_1_s1

    #P(s_ip1=s1|s_i=s2)
    P_s_ip1_s1_s_i_s2 = 0

    #P(s_ip1=s2|s_i=s2) = P(s_ip1=s2|s_i=s2, b_i=0) * P(b_i=0|s_i=s2) + P(s_ip1=s2|s_i=s2, b_i=1) * P(b_i=1|s_i=s2)
    P_s_ip1_s2_s_i_s2 = p_s_ip1_s2_s_i_s2_b_0 * p_b_0_s2 + p_s_ip1_s2_s_i_s2_b_1 * p_b_1_s2

    transitionMat = np.zeros((2, 2)) # from column to row; A_ij = P(s_ip1 = si|s_i=s_j)
    transitionMat[0, 0] = P_s_ip1_s1_s_i_s1
    transitionMat[0, 1] = P_s_ip1_s1_s_i_s2
    transitionMat[1, 0] = P_s_ip1_s2_s_i_s1
    transitionMat[1, 1] = P_s_ip1_s2_s_i_s2

    s_curr = np.argmax(transitionMat[:, s_prev]) 
    sequence_arr_test.append(s_curr)


    # Step 2: Propate one step forward (or retrieve the current state from hardware)
    x_curr = x_prev + x_dot_test[-1] * dt
    x_test.append(x_curr)  



    # Step 3: Compute the predicted velocity 
    x_dot_pred_curr = pos_ds[s_curr].predict(x_curr).T
    x_dot_pred.append(x_dot_pred_curr)
    if i >= 395 and i <=520:
        x_dot_test_curr = np.array((0, 30)).reshape(1, -1) # adding manual perturbation
        x_dot_test.append(x_dot_test_curr)
    else:
        x_dot_test.append(x_dot_pred_curr)



    # Step 4: Detect perturbation and update s_curr
    x_dot_test_curr_norm = x_dot_test[-1]/np.linalg.norm(x_dot_test[-1])
    x_dot_pred_curr_norm = x_dot_pred[-1]/np.linalg.norm(x_dot_pred[-1])

    if np.arccos(np.dot(x_dot_test_curr_norm[0], x_dot_pred_curr_norm[0])) > np.pi/6:
        if c_i !=0:
            c_i -= 1
    else:
        if c_i != 10:
            c_i += 1 
    confidence_arr_test.append(c_i)


    if c_i == 0: # update s_curr based on current state in the event of continuous perturbation
        gmm_prob = []
        for ds in pos_ds:
            gmm_prob.append(ds.damm.totalProb(x_curr))
        s_curr = np.argmax(np.array(gmm_prob))
        sequence_arr_test[-1] = s_curr



fig, ax = plt.subplots()

ax.scatter(x[:, 0], x[:, 1], color='k', s=5, alpha=0.4, label='original data')

# Set up the plot limits
# ax.set_xlim(-50, 10)
# ax.set_ylim(-10, 20)

# Generate data
xx = np.array(x_test)[:, 0, 0]
yy = np.array(x_test)[:, 0, 1]


colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']
labels = np.take(colors, np.array(sequence_arr_test))

# Create line segments
points = np.array([xx, yy]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Initialize LineCollection
lc = LineCollection(segments, linewidth=5)
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

s = 0
ds = pos_ds[s]
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


s = 1
ds = pos_ds[s]
A = ds.A
att = ds.x_att
gamma = ds.damm.logProb(X.T)
for k in np.arange(len(A)):
    if k == 0:
        dx = gamma[k].reshape(1, -1) * (A[k] @ (X - att.T))  # gamma[k].reshape(1, -1): [1, num] dim x num
    else:
        dx +=  gamma[k].reshape(1, -1) * (A[k] @ (X - att.T)) 
u2 = dx[0,:].reshape((plot_sample,plot_sample))
v2 = dx[1,:].reshape((plot_sample,plot_sample))

ax.streamplot(x_mesh,y_mesh,u1,v1, density=3.0, color="black", arrowsize=1.1, arrowstyle="->")
scatter1 = ax.scatter(x_att_1[0, 0], x_att_1[0, 1], s=80, facecolors='none', edgecolors='magenta', linewidths=2)
scatter2 = ax.scatter(x_att_2[0, 0], x_att_2[0, 1], s=80, facecolors='none', edgecolors='magenta', linewidths=2)
scatter2.set_visible(False)





def demo_and_streamline_2():
    for art in ax.get_children():
        if isinstance(art, matplotlib.patches.FancyArrowPatch):
            art.remove()        # Method 1
            # art.set_visible(False) # Method 2
    for art in ax.get_children():
        if isinstance(art, LineCollection) and art is not lc:
            art.remove()    
    ax.streamplot(x_mesh,y_mesh,u2,v2, density=3.0, color="black", arrowsize=1.1, arrowstyle="->")
    scatter1.set_visible(False)
    scatter2.set_visible(True)

def demo_and_streamline_1():
    for art in ax.get_children():
        if isinstance(art, matplotlib.patches.FancyArrowPatch):
            art.remove()        # Method 1
            # art.set_visible(False) # Method 2
    for art in ax.get_children():
        if isinstance(art, LineCollection) and art is not lc:
            art.remove()    
    ax.streamplot(x_mesh,y_mesh,u1,v1, density=3.0, color="black", arrowsize=1.1, arrowstyle="->")
    scatter1.set_visible(True)
    scatter2.set_visible(False)

def init():
    """Initialize the background of the plot."""
    lc.set_segments([])

    return lc,

def update(frame):
    """Update the plot with new data for each frame."""
    if frame!=0 and sequence_arr_test[frame] != sequence_arr_test[frame-1] and sequence_arr_test[frame] == 1:
        demo_and_streamline_2()
    elif sequence_arr_test[frame] != sequence_arr_test[frame-1] and sequence_arr_test[frame] == 0:
        demo_and_streamline_1()

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

# ani.save("animation.gif", writer=PillowWriter(fps=45))

# import matplotlib.animation as animation
# FFwriter = animation.FFMpegWriter(fps=45)
# ani.save('animation.mp4', writer = FFwriter)

# Show the animation
plt.show()

