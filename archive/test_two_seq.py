import numpy as np
import matplotlib.pyplot as plt

from src.util import optimize_tools, load_tools, process_tools
from src.lpvds.src.lpvds_class import lpvds_class

from scipy.stats import multivariate_normal

# %matplotlib tk


'''Load data'''
p_raw, q_raw, t_raw, dt = load_tools.load_npy()
'''Process data'''
p_in, q_in, t_in             = process_tools.pre_process(p_raw, q_raw, t_raw, opt= "savgol")
p_out, q_out                 = process_tools.compute_output(p_in, q_in, t_in)
p_init, q_init, p_att, q_att = process_tools.extract_state(p_in, q_in)
p_in, q_in, p_out, q_out     = process_tools.rollout_list(p_in, q_in, p_out, q_out)


x = p_in
x_dot = p_out
x_att = p_att
x_init = p_init

N1 = 100
sequence_arr = np.zeros((p_in.shape[0], ), dtype=int)
sequence_arr[N1: ] = 1


x_1 = x[sequence_arr==0, :]
x_dot_1 = x_dot[sequence_arr==0, :]


x_2 = x[sequence_arr==1, :]
x_dot_2 = x_dot[sequence_arr==1, :]



colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']
color_mapping = np.take(colors, sequence_arr)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], color=color_mapping[:], alpha= 0.6, s=0.5)
ax.axis('equal')
plt.show()


x_1 = x[sequence_arr==0, :]
x_dot_1 = x_dot[sequence_arr==0, :]

x_2 = x[sequence_arr==1, :]
x_dot_2 = x_dot[sequence_arr==1, :]


x_att_1 = x_1[-1, :] - 0.01 * np.random.rand(3,).reshape(1, -1) # Random initialization
# x_att_2 = x_att - 0.01 * np.random.rand(3,).reshape(1, -1) # Random initialization

x_att_2 = x_att.reshape(1, -1) # Random initialization


T = 15
x_att_list = []
MSE_list = []

for i in range(T):
    P_1 = optimize_tools._optimize_P(x_1-x_att_1, x_dot_1, np.zeros((x_1.shape[0], ), dtype=int))
    A_1 = optimize_tools.optimize_A(x_1, x_dot_1, x_att_1, gamma=np.ones((1, x_1.shape[0]), dtype=int), P=P_1)
    x_att_1 = optimize_tools.optimize_att(x_1, x_dot_1, A_1)

    # P_2 = optimize_tools._optimize_P(x_2-x_att_2, x_dot_2, np.zeros((x_2.shape[0], ), dtype=int))
    # A_2 = optimize_tools.optimize_A(x_2, x_dot_2, x_att_2, gamma=np.ones((1, x_2.shape[0]), dtype=int), P=P_2)
    # x_att_2 = optimize_tools.optimize_att(x_2, x_dot_2, A_2)

    # x_att_2 = optimize_tools.optimize_att(x_2, x_dot_2, A_2)
    if i%100==0:
        print(i)

    x_dot_1_pred = A_1[0] @ (x_1 - x_att_1).T
    MSE_list.append(np.sum(np.linalg.norm(x_dot_1_pred-x_dot_1.T, axis=0)))


pos_ds_1 = lpvds_class(x_1, x_dot_1, x_att_1)
pos_ds_1.begin()

pos_ds_2 = lpvds_class(x_2, x_dot_2, x_att_2)
pos_ds_2.begin()

# A_2 = optimize_tools.optimize_A(x_2, x_dot_2, x_att_2)


colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']
color_mapping = np.take(colors, sequence_arr)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], color=color_mapping[:], alpha= 0.6, s=0.5)
ax.scatter(x_att_1[:, 0], x_att_1[:, 1], x_att_1[:, 2], color='k', alpha= 0.6, s=0.5)
ax.scatter(x_att_2[:, 0], x_att_2[:, 1], x_att_2[:, 2], color='k', alpha= 0.6, s=0.5)

ax.axis('equal')
plt.show()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot()
ax.scatter(np.arange(T), np.array(MSE_list))
plt.show()


# # evaluate results
x_test_list_1 = []
x_test_list_2 = []

for x_0 in x_init:
    x_test_list_1.append(pos_ds_1.sim(x_0.reshape(1, -1), dt=0.01))
    x_test_list_2.append(pos_ds_2.sim(x_2[0, :].reshape(1, -1), dt=0.01))


# plot results
from src.lpvds.src.util import plot_tools
plot_tools.plot_ds_3d(x, x_test_list_1)
plot_tools.plot_ds_3d(x, x_test_list_2)
plt.show()

