import autograd.numpy as np
import scipy
import random
from gradient_descent import als_gradient_descent_U_step, als_gradient_descent_V_step
from misc import get_k_rank_mxn_matrix,random_binary_mask

ALS_MAX_ITERATIONS = 50

# #TODO
# def als_for_implicit_feedback(obs,mask,rank, alpha_factor = 10):
#     # Algorithm from Collaborative Filtering for Implicit Feedback Datasets (2008)
#     # https://ieeexplore.ieee.org/document/4781121
#     Y = obs
#     r = mask
#     c = 1 + mask * alpha_factor




def als(obs, mask, args):
    # algorithm from "Low-rank Matrix Completion using Alternating Minimization (2012)"
    # https://arxiv.org/pdf/1212.0467.pdf
    rank = args.rank
    print('\nSTARTING TRAINING ..... \n')
    m,n = obs.shape
    U_0 = [random.random() for i in range(m*rank)]
    U_0 = np.array(U_0).reshape(m,rank)
    V_0 = [random.random() for i in range(n*rank)]
    V_0 = np.array(V_0).reshape(n,rank)

    U_new,V_new=np.copy(U_0),np.copy(V_0)

    for idx in range(args.max_ALS_steps):
        print('\t TRAINING: ALS STEP: {0} / {1}'.format(idx,args.max_ALS_steps))
        U_new = als_gradient_descent_U_step(obs, mask, U_new, V_new,args)
        V_new = als_gradient_descent_V_step(obs, mask, U_new, V_new,args)

    return (U_new,V_new, U_0, V_0)

def matrix_completion(m,n,rank,observed_fraction):
    fully_observed, rank = get_k_rank_mxn_matrix(m,n,rank)
    print('rank',rank)
    mask = random_binary_mask(m,n,observed_fraction)
    obs = fully_observed* mask
    als(obs,mask,rank)

if __name__ == "__main__":
    matrix_completion(10000,1000,20,0.7)





