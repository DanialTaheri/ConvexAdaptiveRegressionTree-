import numpy as np

def concaveEval(alpha, beta, x):
# This function takes the alphas and betas exported by convexTree, fits a
# function and then evaluates it at x, returning y
    alpha_arr = np.array(alpha)
    beta_arr = np.array(beta)
    print("Number of classes", beta_arr.shape[0])
    y_pred = np.zeros((x.shape[0], beta_arr.shape[0]))
    for i in range(beta_arr.shape[0]):
        y_pred[:, i] = np.dot(x, beta_arr[i, :].T) + alpha_arr[i]
    

    return y_pred.min(axis = 1)