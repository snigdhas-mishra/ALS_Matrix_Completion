import autograd.numpy as np
from numpy.linalg import matrix_rank

def validation_objective(observed,predicted,mask):
    loss = np.sum(mask * ((observed - predicted) ** 2)) / np.sum(mask)
    return loss

def validation(val_obs, val_mask, U, V):
    return validation_objective(val_obs,np.dot(U,V.T),val_mask)

def get_k_rank_mxn_matrix(m,n,k):

    U = np.random.rand(m,k)
    V = np.random.rand(n,k)

    P = np.dot(U,V.T)

    rank_value = matrix_rank(P)
    return P, rank_value

def random_binary_mask(m,n, observed_fraction):
    num_of_ones = int(observed_fraction * m*n)
    u = np.zeros((m*n))
    u[:num_of_ones] = 1.0
    np.random.shuffle(u)
    u = u.reshape(m,n)
    return u

def testing_unit_functions():
    mat, rank = get_k_rank_mxn_matrix(1000,100, 10 )
    print(rank, mat.shape)
    print(random_binary_mask(3,2,0.5))


if __name__=="__main__":
    testing_unit_functions()