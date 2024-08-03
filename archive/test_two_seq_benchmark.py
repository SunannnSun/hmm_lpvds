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



pos_ds_1 = lpvds_class(x, x_dot, x_att)
pos_ds_1.begin()



x_test_list = []
for x_0 in x_init:
    x_test_list.append(pos_ds_1.sim(x[0, :].reshape(1, -1), dt=0.01))


# plot results
from src.lpvds.src.util import plot_tools
plot_tools.plot_ds_3d(x, x_test_list)
plt.show()
