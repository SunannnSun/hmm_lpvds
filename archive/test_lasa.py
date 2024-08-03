import numpy as np
# from src.util import optimize_tools, load_tools, process_tools
from src.util import optimize_tools
from src.lpvds.src.lpvds_class import lpvds_class
from src.lpvds.src.dsopt.dsopt_class import dsopt_class

from src.lpvds.src.util import load_tools, plot_tools

import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


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


N1 = 200
sequence_arr = np.zeros((x.shape[0], ), dtype=int)
sequence_arr[N1: ] = 1

L = 1
for l in range(L):
    x_1 = x[sequence_arr==0, :]
    x_dot_1 = x_dot[sequence_arr==0, :]

    x_2 = x[sequence_arr==1, :]
    x_dot_2 = x_dot[sequence_arr==1, :]

    pos_ds_1 = lpvds_class(x_1, x_dot_1, x_att)
    pos_ds_2 = lpvds_class(x_2, x_dot_2, x_att)

    pos_ds_1._cluster()
    # pos_ds_2._cluster()

    plot_tools.plot_gmm(x_1, pos_ds_1.assignment_arr, pos_ds_1.damm)
    # plot_tools.plot_gmm(x_2, pos_ds_2.assignment_arr, pos_ds_2.damm)

    gamma_1 = pos_ds_1.gamma
    # gamma_2 = pos_ds_2.gamma
    # pos_ds_2.K = 1
    # pos_ds_2.gamma = np.zeros((1, x_2.shape[0]))

    """Initialize"""
    # A_1 = [-1 * np.eye(2)] * gamma_1.shape[0]
    # A_2 = [-1 * np.eye(2)] * gamma_2.shape[0]
    x_att_1 = x_1[-1, :] - 0.01 * np.random.rand(2,).reshape(1, -1) # Random initialization
    x_att_2 = x_att - 0.01 * np.random.rand(2,).reshape(1, -1) # Random initialization


    T = 10
    x_att_list = []
    MSE_list = []
    for i in range(T):

        # x_att_1 = optimize_tools.optimize_att(x_1, x_dot_1, A_1, gamma=gamma_1)
        P_1 = optimize_tools._optimize_P(x_1-x_att_1, x_dot_1, pos_ds_1.assignment_arr)
        A_1 = optimize_tools.optimize_A(x_1, x_dot_1, x_att_1, gamma=gamma_1, P=P_1)

        # P_2 = optimize_tools._optimize_P(x_2-x_att_2, x_dot_2, pos_ds_2.assignment_arr)
        # A_2 = optimize_tools.optimize_A(x_2, x_dot_2, x_att_2, gamma=gamma_2, P=P_2)
        # x_att_2 = optimize_tools.optimize_att(x_2, x_dot_2, A_2, gamma=gamma_2)
        pos_ds_1.A = A_1
        pos_ds_1.x_att = x_att_1
        MSE_list.append(pos_ds_1.evaluate())

        print('Learned attractor: ', x_att_1)
        x_att_list.append(x_att_1[0])

    # A_2 = optimize_tools.optimize_A(x_2, x_dot_2, x_att_2, gamma=gamma_2)
    # ds_opt = dsopt_class(x_2, x_dot_2, x_att_2, gamma_2, np.argmax(gamma_2, axis=0))
    # A_2 = ds_opt.begin()
    pos_ds_2.begin()
    A_2 = pos_ds_2.A


    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot()
    ax.scatter(np.arange(T), np.array(MSE_list))
    plt.show()


    x_att_arr = np.array(x_att_list)
    print('Given attractor: ', x_att)


    colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']
    color_mapping = np.take(colors, sequence_arr)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot()
    ax.scatter(x[:, 0], x[:, 1], color=color_mapping[:], alpha= 0.6, s=0.5)
    ax.scatter(x_att_1[:, 0], x_att_1[:, 1], color='k', alpha= 0.6, s=0.5)
    ax.scatter(x_att_2[:, 0], x_att_2[:, 1], color='k', alpha= 0.6, s=0.5)

    ax.axis('equal')
    # plt.show()

    pos_ds_1.A = A_1
    pos_ds_1.x_att = x_att_1

    pos_ds_2.A = A_2
    pos_ds_2.x_att = x_att_2


    # # # evaluate results
    # x_test_list_1 = []
    # x_test_list_2 = []
    # for x_0 in x_init:
    #     x_test_list_1.append(pos_ds_1.sim(x_0.reshape(1, -1), dt=0.01))
    #     x_test_list_2.append(pos_ds_2.sim(x_2[0, :].reshape(1, -1), dt=0.01))


    # # plot results
    # from src.lpvds.src.util import plot_tools
    # plot_tools.plot_ds_3d(x, x_test_list_1)
    # plot_tools.plot_ds_3d(x, x_test_list_2)
    # plt.show()


    n_S = 2 # number of S
    M = x.shape[0]

    alpha = np.zeros((M, n_S))

    # Termination probability P(b=1|s, x)

    thld = 0.04
    # for i in range(x.shape[0]):
    transitionMat = np.zeros((x.shape[0]-1, 2, 2)) # from column to row; A_ij = P(s_ip1 = si|s_i=s_j)

    for i in np.arange(1, x.shape[0]):
    # for i in [1]:
        x_i = x[i, :]

        #P(b_i=1|s_i=s1)
        p_b_1_s1 = 1 if np.linalg.norm(x_i - x_2[0, :]) < thld else 0
        #P(b_i=0|s_i=s1)
        p_b_0_s1 = 1 - p_b_1_s1

        #P(b_i=1|s_i=s2)
        p_b_1_s2 = 1 if np.linalg.norm(x_i -x_2[-1, :]) < thld else 0
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

        #P(s_ip1=s2|s_i=s1) = 0 * x + 1 * y
        P_s_ip1_s2_s_i_s1 =  p_s_ip1_s2_s_i_s1_b_0 * p_b_0_s1 + p_s_ip1_s2_s_i_s1_b_1 * p_b_1_s1
        # print(np.linalg.norm(x_i - x_att_2))


        #P(s_ip1=s1|s_i=s2)
        P_s_ip1_s1_s_i_s2 = 1


        #P(s_ip1=s2|s_i=s2) = P(s_ip1=s2|s_i=s2, b_i=0) * P(b_i=0|s_i=s2) + P(s_ip1=s2|s_i=s2, b_i=1) * P(b_i=1|s_i=s2)
        P_s_ip1_s2_s_i_s2 = p_s_ip1_s2_s_i_s2_b_0 * p_b_0_s2 + p_s_ip1_s2_s_i_s2_b_1 * p_b_1_s2


        transitionMat[i-1, 0, 0] = P_s_ip1_s1_s_i_s1
        transitionMat[i-1, 0, 1] = P_s_ip1_s1_s_i_s2
        transitionMat[i-1, 1, 0] = P_s_ip1_s2_s_i_s1
        transitionMat[i-1, 1, 1] = P_s_ip1_s2_s_i_s2

    def adjust_cov(cov, tot_scale_fact=2, rel_scale_fact=0.15):

        eigenvalues, eigenvectors = np.linalg.eig(cov)

        idxs = eigenvalues.argsort()
        inverse_idxs = np.zeros((idxs.shape[0]), dtype=int)
        for index, element in enumerate(idxs):
            inverse_idxs[element] = index

        eigenvalues_sorted  = np.sort(eigenvalues)
        cov_ratio = eigenvalues_sorted[0]/eigenvalues_sorted[1]
        if cov_ratio < rel_scale_fact:
            lambda_2 = eigenvalues_sorted[1]
            lambda_1 = eigenvalues_sorted[0] + lambda_2 * (rel_scale_fact - cov_ratio)

            lambdas = np.array([lambda_1, lambda_2])

            L = np.diag(lambdas[inverse_idxs]) * tot_scale_fact
        else:
            L = np.diag(eigenvalues) * tot_scale_fact

        Sigma = eigenvectors @ L @ eigenvectors.T

        return Sigma


    mu_1 = np.mean(x_1, axis=0)
    Sigma_1 = np.cov(x_1.T)
    Sigma_1 = adjust_cov(Sigma_1)

    mu_2 =  np.mean(x_2, axis=0)
    Sigma_2 = np.cov(x_2.T)
    Sigma_2 = adjust_cov(Sigma_2)

    cov_em_1 = (x_dot_1.T - (A_1[0] @ (x_1-x_att_1).T)) @ (x_dot_1.T - (A_1[0] @ (x_1-x_att_1).T)).T / x_dot_1.shape[0]

    # cov_em_2 = np.cov((x_dot_2.T - (A_2[0] @ (x_2-x_att_2).T)))

    cov_em_2 =(x_dot_2.T - (A_2[0] @ (x_2-x_att_2).T)) @ (x_dot_2.T - (A_2[0] @ (x_2-x_att_2).T)).T / (x_dot_2.shape[0] - 1)

    emissionMat = np.zeros((x.shape[0], 2)) 
    # for i in [0, 1]:
    for i in range(x.shape[0]):
        x_i = x[i, :]
        x_dot_i = x_dot[i, :]

        #P(x_ip1|s_ip1=s1) 
        P_x_ip1_s_ips_s1 = multivariate_normal.logpdf(x_i, mean=mu_1, cov=Sigma_1, allow_singular=True)

        #P(x_ip1|s_ip1=s2) 
        P_x_ip1_s_ips_s2 = multivariate_normal.logpdf(x_i, mean=mu_2, cov=Sigma_2, allow_singular=True)

        #P(x_dot_ip1|x_ip1, s_ip1=s1)
        mu_em_1 = A_1[0] @ (x_i.reshape(1, -1) - x_att_1).T
        P_x_dot_ip1_x_ip1_s_ip1_s1 =  multivariate_normal.logpdf(x_dot_i, mean=mu_em_1[:, 0], cov=cov_em_1, allow_singular=True)

        #P(x_dot_ip1|x_ip1, s_ip1=s2)
        mu_em_2 = A_2[0] @ (x_i.reshape(1, -1) - x_att_2).T
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


    # alpha_1(s_1=s1) = P(x_0, x_dot_0, x_1, x_dot_1, s_1=s1) 
    #                 = P(x_1, x_dot_1|s_1=s1) * (P(x_0, x_dot_0, s_0=s1)*P(s_1=s1|s_0=s1)+P(x_0, x_dot_0, s_0=s2)*P(s_1=s1|s_0=s2))


    # alpha_0(s_0=s1) = P(x_0, x_dot_0, s_0=s1) = P(x_0, x_dot_0|s_0=s1) * P(s_0=s1)
    if l == L-1:
        a=1
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


        b_1 = np.log(alpha[i-1,0] * transitionMat[i-1, 0, 0] + alpha[i-1,1] * transitionMat[i-1, 0, 1])
        alpha[i, 0] = emissionMat[i, 0] 

        # alpha_1(s_1=s2) = P(x_0, x_dot_0, x_1, x_dot_1, s_1=s2) 
        #                 = P(x_1, x_dot_1|s_1=s1) * (P(x_0, x_dot_0, s_0=s1)*P(s_1=s2|s_0=s1)+P(x_0, x_dot_0, s_0=s2)*P(s_1=s2|s_0=s2))
        # alpha[i, 1] = emissionMat[i, 1] + np.log(alpha[i-1,0] * transitionMat[i-1, 1, 0] + alpha[i-1,1] * transitionMat[i-1, 1, 1])
        b_2 = np.log(alpha[i-1,0] * transitionMat[i-1, 1, 0] + alpha[i-1,1] * transitionMat[i-1, 1, 1])
        alpha[i, 1] = emissionMat[i, 1] 


        logProb = alpha[i, :].reshape(-1, 1)
        maxPostLogProb = np.max(logProb, axis=0, keepdims=True)
        expProb = np.exp(logProb - np.tile(maxPostLogProb, (2, 1)))
        postProb = expProb / np.sum(expProb, axis = 0, keepdims=True)
    
        alpha[i, :] = postProb[:, 0]


    # sequence_arr = np.argmax(alpha, axis = 1)


    # x_1 = x[sequence_arr==0, :]
    # x_dot_1 = x_dot[sequence_arr==0, :]


    # x_2 = x[sequence_arr==1, :]
    # x_dot_2 = x_dot[sequence_arr==1, :]

    print(l)

    plot_tools.plot_gmm(x_2, pos_ds_2.assignment_arr, pos_ds_2.damm)

    # # evaluate results
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

    sequence_arr = np.argmax(alpha, axis = 1)


    # x_1 = x[sequence_arr==0, :]
    # x_dot_1 = x_dot[sequence_arr==0, :]


    # x_2 = x[sequence_arr==1, :]
    # x_dot_2 = x_dot[sequence_arr==1, :]


    # colors = ["r", "g", "b", "k", 'c', 'm', 'y', 'crimson', 'lime']
    # color_mapping = np.take(colors, sequence_arr)
    # fig = plt.figure(figsize=(12, 10))
    # ax = fig.add_subplot()
    # ax.scatter(x[:, 0], x[:, 1], color=color_mapping[:], alpha= 0.6, s=0.5)
    # ax.axis('equal')
    # plt.show()

plot_tools.plot_gmm(x_2, pos_ds_2.assignment_arr, pos_ds_2.damm)

# # evaluate results
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