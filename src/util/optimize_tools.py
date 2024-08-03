import cvxpy as cp
import numpy as np




def _optimize_P(x_sh, x_dot, assignment_arr):
    """Fast/convex Lyapunov function learning by Tianyu"""
    # x = self.x_sh
    # x_dot = self.x_dot
    # assignment_arr = self.assignment_arr

    x = x_sh
    K = np.max(assignment_arr) + 1
    N = x.shape[1]


    x_mean_vec = []
    mean_vec = []
    for k in range(K):
        x_k = x[assignment_arr==k,:]
        x_dot_k = x_dot[assignment_arr==k, :]

        x_mean_k = np.mean(x_k, axis=0)
        x_dot_mean_k = np.mean(x_dot_k, axis=0)
        x_dot_mean_k = (x_dot_mean_k / np.linalg.norm(x_dot_mean_k))

        x_mean_vec.append(x_mean_k)
        mean_vec.append(x_dot_mean_k)
        
    x_mean_vec = np.array(x_mean_vec)
    mean_vec = np.array(mean_vec)

    P = cp.Variable((N, N), symmetric=True)

    constraints = [P >> 1e-3] # set a margin to avoid computational issue
    objective = 0
    # for xi, vi in zip(x_mean_vec, mean_vec):
    #     projection = vi @ P @ xi
    #     violation = cp.pos(projection)
    #     objective += violation

    projections = cp.sum(cp.multiply(x_mean_vec @ P, mean_vec), axis=1)
    # projections = cp.sum(cp.multiply(x @ P, x_dot), axis=1)
    violations  = cp.pos(projections)
    objective   = cp.sum(violations)

    objective = cp.Minimize(objective)
    prob = cp.Problem(objective, constraints)

    prob.solve(verbose=False)

    P_opt = P.value
    # print("Optimal Matrix P:\n", P_opt)

    return P_opt




def optimize_A(x, x_dot, x_att, **argv):
    M, N = x.shape
    x_sh = x - x_att


    if len(argv) == 0:
        gamma = np.ones((M, )).reshape(1, -1)
        P = np.eye(N)
    elif len(argv) == 1:
        gamma = argv['gamma']
        P = np.eye(N)
    else:
        gamma = argv['gamma']
        P = argv['P']


    K = gamma.shape[0]

    # Define variables and constraints
    A_vars = []
    Q_vars = []
    constraints = []
    max_norm = 5
    for k in range(K):
        A_vars.append(cp.Variable((N, N)))
        Q_vars.append(cp.Variable((N, N), symmetric=True))

        epi = 0.001
        epi = epi * -np.eye(N)

        constraints += [A_vars[k].T @ P + P @ A_vars[k] == Q_vars[k]]
        constraints += [Q_vars[k] << epi]
        # constraints += [cp.norm(A_vars[k], 'fro') <= max_norm]


    for k in range(K):
        x_dot_pred_k = A_vars[k] @ x_sh.T
        if k == 0:
            x_dot_pred  = cp.multiply(np.tile(gamma[k, :], (N, 1)), x_dot_pred_k)
        else:
            x_dot_pred += cp.multiply(np.tile(gamma[k, :], (N, 1)), x_dot_pred_k)


    Objective = cp.norm(x_dot_pred.T-x_dot, 'fro')

    prob = cp.Problem(cp.Minimize(Objective), constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    A_res = np.zeros((K, N, N))
    for k in range(K):
        A_res[k, :, :] = A_vars[k].value
    
    return A_res



def optimize_att(x, x_dot, A, **argv):
    M, N = x.shape
    # x_sh = x - x_att
    

    if len(argv) == 0:
        gamma = np.ones((M, )).reshape(1, -1)
        P = np.eye(N)
    elif len(argv) == 1:
        gamma = argv['gamma']
        P = np.eye(N)
    else:
        gamma = argv['gamma']
        P = argv['P']



    # if len(argv) == 0:
    #     gamma = np.ones((M, )).reshape(1, -1)
    # else:
    #     gamma = argv['gamma']

    K = gamma.shape[0]

    # Define variables and constraints
    constraints = []

    x_att = cp.Variable((1, N))
    for k in range(K):
        x_dot_pred_k = A[k] @ (x-x_att).T
        if k == 0:
            x_dot_pred  = cp.multiply(np.tile(gamma[k, :], (N, 1)), x_dot_pred_k)
        else:
            x_dot_pred += cp.multiply(np.tile(gamma[k, :], (N, 1)), x_dot_pred_k)


    Objective = cp.norm(x_dot_pred.T-x_dot, 'fro')

    prob = cp.Problem(cp.Minimize(Objective), constraints)
    prob.solve(solver=cp.MOSEK, verbose=False)

    x_att_ = np.zeros((N, ))
    x_att_ = x_att.value
    # for k in range(K):
    #     x_att[k, :, :] = A_vars[k].value
        # print(A_vars[k].value)
    
    return x_att_