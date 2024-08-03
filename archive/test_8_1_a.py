import numpy as np
# from src.util import optimize_tools, load_tools, process_tools
from src.util import optimize_tools
from src.lpvds.src.lpvds_class import lpvds_class
from src.lpvds.src.dsopt.dsopt_class import dsopt_class

from src.lpvds.src.util import load_tools, plot_tools

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


N1 = 500
N_att = 50

sequence_arr = np.zeros((x.shape[0], ), dtype=int)
sequence_arr[N1: ] = 1

L = 1
for l in range(L):
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

    MSE_list = []
    MSE_list.append(pos_ds_1.evaluate())
    pos_ds_2.evaluate()
    
    # plot_tools.plot_gmm(x_1, pos_ds_1.assignment_arr, pos_ds_1.damm)
    # plot_tools.plot_gmm(x_2, pos_ds_2.assignment_arr, pos_ds_2.damm)

    n_S = 2 # number of S
    M = x.shape[0]

    alpha = np.zeros((M, n_S))

    mu_1 = np.mean(x_1, axis=0)
    Sigma_1 = np.cov(x_1.T)

    mu_2 =  np.mean(x_2, axis=0)
    Sigma_2 = np.cov(x_2.T)

    cov_em_1 = (x_dot_1.T - pos_ds_1.x_dot_pred) @ (x_dot_1.T - pos_ds_1.x_dot_pred).T / x_dot_1.shape[0]
    cov_em_2 = (x_dot_2.T - pos_ds_2.x_dot_pred) @ (x_dot_2.T - pos_ds_2.x_dot_pred).T / x_dot_2.shape[0]

    emissionMat = np.zeros((x.shape[0], 2)) 
    for i in range(x.shape[0]):
        x_i = x[i, :]
        x_dot_i = x_dot[i, :]

        #P(x_ip1|s_ip1=s1) 
        P_x_ip1_s_ips_s1 = multivariate_normal.logpdf(x_i, mean=mu_1, cov=Sigma_1, allow_singular=True)

        #P(x_ip1|s_ip1=s2) 
        P_x_ip1_s_ips_s2 = multivariate_normal.logpdf(x_i, mean=mu_2, cov=Sigma_2, allow_singular=True)

        #P(x_dot_ip1|x_ip1, s_ip1=s1)
        mu_em_1 = pos_ds_1.predict(x_i.reshape(1, -1))
        P_x_dot_ip1_x_ip1_s_ip1_s1 =  multivariate_normal.logpdf(x_dot_i, mean=mu_em_1[:, 0], cov=cov_em_1, allow_singular=True)

        #P(x_dot_ip1|x_ip1, s_ip1=s2)
        mu_em_2 =pos_ds_2.predict(x_i.reshape(1, -1))
        P_x_dot_ip1_x_ip1_s_ip1_s2 =  multivariate_normal.logpdf(x_dot_i, mean=mu_em_2[:, 0], cov=cov_em_2, allow_singular=True)


        #P(x_ip1, x_dot_ip1|s_ip1=s1) = P(x_dot_ip1|x_ip1, s_ip1=s1) * P(x_ip1|s_ip1=s1) 
        # P_x_ip1_x_dot_ip1_s_ip1_s1 = P_x_dot_ip1_x_ip1_s_ip1_s1 * P_x_ip1_s_ips_s1
        P_x_ip1_x_dot_ip1_s_ip1_s1 = P_x_dot_ip1_x_ip1_s_ip1_s1 + P_x_ip1_s_ips_s1

        #P(x_ip1, x_dot_ip1|s_ip1=s2) = P(x_dot_ip1|x_ip1, s_ip1=s2) * P(x_ip1|s_ip1=s2) 
        # P_x_ip1_x_dot_ip1_s_ip1_s2 = P_x_dot_ip1_x_ip1_s_ip1_s2 * P_x_ip1_s_ips_s2
        P_x_ip1_x_dot_ip1_s_ip1_s2 = P_x_dot_ip1_x_ip1_s_ip1_s2 + P_x_ip1_s_ips_s2

        emissionMat[i, 0] = P_x_ip1_x_dot_ip1_s_ip1_s1
        emissionMat[i, 1] = P_x_ip1_x_dot_ip1_s_ip1_s2

    # print(emissionMat)


#     # alpha_1(s_1=s1) = P(x_0, x_dot_0, x_1, x_dot_1, s_1=s1) 
#     #                 = P(x_1, x_dot_1|s_1=s1) * (P(x_0, x_dot_0, s_0=s1)*P(s_1=s1|s_0=s1)+P(x_0, x_dot_0, s_0=s2)*P(s_1=s1|s_0=s2))


    # alpha_0(s_0=s1) = P(x_0, x_dot_0, s_0=s1) = P(x_0, x_dot_0|s_0=s1) * P(s_0=s1)
    # if l == L-1:
    #     a=1
    alpha[0, 0] = emissionMat[0, 0] + np.log(1/2)

    # alpha_0(s_0=s2) = P(x_0, x_dot_0, s_0=s2) = P(x_0, x_dot_0|s_0=s2) * P(s_0=s2)
    alpha[0, 1] = emissionMat[0, 1] + np.log(1/2)

    # normalize

    logProb = alpha[0, :].reshape(-1, 1)
    maxPostLogProb = np.max(logProb, axis=0, keepdims=True)
    expProb = np.exp(logProb - np.tile(maxPostLogProb, (2, 1)))
    postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)

    alpha[0, :] = postProb[:, 0]


    for i in np.arange(1, alpha.shape[0]):
    # for i in [1]:
        # alpha[1, 0] = P_x_ip1_x_dot_ip1_s_ip1_s1 * (alpha[0,0] * P_s_ip1_s1_s_i_s1 + alpha[0,1] * P_s_ip1_s1_s_i_s2)
        # alpha[i, 0] = emissionMat[i, 0] + np.log(alpha[i-1,0] * transitionMat[i-1, 0, 0] + alpha[i-1,1] * transitionMat[i-1, 0, 1])


        # b_1 = np.log(alpha[i-1,0] * transitionMat[i-1, 0, 0] + alpha[i-1,1] * transitionMat[i-1, 0, 1])
        alpha[i, 0] = emissionMat[i, 0] 

        # alpha_1(s_1=s2) = P(x_0, x_dot_0, x_1, x_dot_1, s_1=s2) 
        #                 = P(x_1, x_dot_1|s_1=s1) * (P(x_0, x_dot_0, s_0=s1)*P(s_1=s2|s_0=s1)+P(x_0, x_dot_0, s_0=s2)*P(s_1=s2|s_0=s2))
        # alpha[i, 1] = emissionMat[i, 1] + np.log(alpha[i-1,0] * transitionMat[i-1, 1, 0] + alpha[i-1,1] * transitionMat[i-1, 1, 1])
        # b_2 = np.log(alpha[i-1,0] * transitionMat[i-1, 1, 0] + alpha[i-1,1] * transitionMat[i-1, 1, 1])
        alpha[i, 1] = emissionMat[i, 1] 


        logProb = alpha[i, :].reshape(-1, 1)
        maxPostLogProb = np.max(logProb, axis=0, keepdims=True)
        expProb = np.exp(logProb - np.tile(maxPostLogProb, (2, 1)))
        postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)
    
        alpha[i, :] = postProb[:, 0]


    sequence_arr = np.argmax(alpha, axis = 1)

    print(time.time()-T_start)


    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']
    color_mapping = np.take(colors, sequence_arr)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot()
    ax.scatter(x[:, 0], x[:, 1], color=color_mapping[:], alpha= 0.6, s=0.5)
    ax.scatter(x_att_1[:, 0], x_att_1[:, 1], color='k', alpha= 0.6, s=0.5)
    ax.scatter(x_att_2[:, 0], x_att_2[:, 1], color='k', alpha= 0.6, s=0.5)
    ax.axis('equal')
    plt.show()
    
#     # x_1 = x[sequence_arr==0, :]
#     # x_dot_1 = x_dot[sequence_arr==0, :]


#     # x_2 = x[sequence_arr==1, :]
#     # x_dot_2 = x_dot[sequence_arr==1, :]

#     print(l)

#     plot_tools.plot_gmm(x_2, pos_ds_2.assignment_arr, pos_ds_2.damm)

#     # # evaluate results
#     x_test_list_1 = []
#     x_test_list_2 = []

#     for x_0 in x_init:
#         x_test_list_1.append(pos_ds_1.sim(x_0.reshape(1, -1), dt=0.01))
#         x_test_list_2.append(pos_ds_2.sim(x_2[0, :].reshape(1, -1), dt=0.01))


#     # plot results
#     from src.lpvds.src.util import plot_tools
#     plot_tools.plot_ds_2d(x, x_test_list_1, pos_ds_1)
#     plot_tools.plot_ds_2d(x, x_test_list_2, pos_ds_2)
#     plt.show()

#     sequence_arr = np.argmax(alpha, axis = 1)


#     # x_1 = x[sequence_arr==0, :]
#     # x_dot_1 = x_dot[sequence_arr==0, :]


#     # x_2 = x[sequence_arr==1, :]
#     # x_dot_2 = x_dot[sequence_arr==1, :]


#     # colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']
#     # color_mapping = np.take(colors, sequence_arr)
#     # fig = plt.figure(figsize=(12, 10))
#     # ax = fig.add_subplot()
#     # ax.scatter(x[:, 0], x[:, 1], color=color_mapping[:], alpha= 0.6, s=0.5)
#     # ax.axis('equal')
#     # plt.show()

# plot_tools.plot_gmm(x_2, pos_ds_2.assignment_arr, pos_ds_2.damm)

# # # evaluate results
x_test_list_1 = []
x_test_list_2 = []


for x_0 in x_init:
    x_test_list_1.append(pos_ds_1.sim(x_0.reshape(1, -1), dt=0.01))
    x_test_list_2.append(pos_ds_2.sim(x_2[0, :].reshape(1, -1), dt=0.01))


# plot results
from src.lpvds.src.util import plot_tools
plot_tools.plot_ds_2d(x, x_test_list_1, pos_ds_1)
plot_tools.plot_ds_2d(x, x_test_list_2, pos_ds_2)
plt.show()