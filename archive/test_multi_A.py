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

x = p_in[:100, :]
x_dot = p_out[:100, :]
# x_att = p_att
x_att = p_in[-1, :]

pos_ds = lpvds_class(x, x_dot, x_att)
pos_ds._cluster()
gamma = pos_ds.gamma

"""Initialize"""
x_att = p_att - 0.01 * np.random.rand(3,).reshape(1, -1) # Random initialization
# A = [-1 * np.eye(3)] * gamma.shape[0]


T = 100
x_att_list = []
for i in range(T):
    A = optimize_tools.optimize_A(x, x_dot, x_att, gamma=gamma)
    x_att = optimize_tools.optimize_att(x, x_dot, A, gamma=gamma)

    print('Learned attractor: ', x_att)
    x_att_list.append(x_att[0])

x_att_arr = np.array(x_att_list)
print('Given attractor: ', p_att)

pos_ds.A = A


# evaluate results
x_test_list = []
for x_0 in p_init:
    x_test_list.append(pos_ds.sim(x_0.reshape(1, -1), dt=0.01))


# plot results
from src.lpvds.src.util import plot_tools
plot_tools.plot_ds_3d(x, x_test_list)
plt.show()