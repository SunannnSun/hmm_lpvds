import numpy as np
from src.util import optimize_tools, load_tools, process_tools
from src.lpvds.src.lpvds_class import lpvds_class
import matplotlib.pyplot as plt


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

x = p_in
x_dot = p_out

N1 = 50
sequence_arr = np.zeros((p_in.shape[0], ), dtype=int)
sequence_arr[N1: ] = 1

x_1 = x[sequence_arr==0, :]
x_dot_1 = x_dot[sequence_arr==0, :]


x_2 = x[sequence_arr==1, :]
x_dot_2 = x_dot[sequence_arr==1, :]


pos_ds_1 = lpvds_class(x_1, x_dot_1, x_att)
pos_ds_2 = lpvds_class(x_2, x_dot_2, x_att)

pos_ds_1._cluster()
pos_ds_2._cluster()

gamma_1 = pos_ds_1.gamma
gamma_2 = pos_ds_2.gamma


"""Initialize"""
# x_att = p_att - 0.01 * np.random.rand(3,).reshape(1, -1) # Random initialization
A_1 = [-1 * np.eye(3)] * gamma_1.shape[0]
A_2 = [-1 * np.eye(3)] * gamma_2.shape[0]


T = 100
x_att_list = []
for i in range(T):
    x_att_1 = optimize_tools.optimize_att(x_1, x_dot_1, A_1, gamma=gamma_1)
    A_1 = optimize_tools.optimize_A(x_1, x_dot_1, x_att_1, gamma=gamma_1)

    x_att_2 = optimize_tools.optimize_att(x_2, x_dot_2, A_2, gamma=gamma_2)
    A_2 = optimize_tools.optimize_A(x_2, x_dot_2, x_att_2, gamma=gamma_2)

    # print('Learned attractor: ', x_att_1)
    x_att_list.append(x_att_1[0])

x_att_arr = np.array(x_att_list)
print('Given attractor: ', p_att)



n_S = 2 # number of S
M = p_in.shape[0]

alpha = np.zeros((M, n_S))





















"""
colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']
color_mapping = np.take(colors, sequence_arr)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], color=color_mapping[:], alpha= 0.6, s=0.5)
ax.scatter(x_att_1[:, 0], x_att_1[:, 1], x_att_1[:, 2], color='k', alpha= 0.6, s=0.5)
ax.scatter(x_att_2[:, 0], x_att_2[:, 1], x_att_2[:, 2], color='k', alpha= 0.6, s=0.5)

ax.axis('equal')
plt.show()



pos_ds_1.A = A_1
pos_ds_1.x_att = x_att_1


pos_ds_2.A = A_2
pos_ds_2.x_att = x_att_2


# # evaluate results
x_test_list = []
for x_0 in p_init:
    x_test_list.append(pos_ds_1.sim(x_0.reshape(1, -1), dt=0.01))
    # x_test_list.append(pos_ds_2.sim(x_2[0, :].reshape(1, -1), dt=0.01))


# plot results
from src.lpvds.src.util import plot_tools
plot_tools.plot_ds_3d(x, x_test_list)
plt.show()
"""