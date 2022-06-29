import autograd.numpy as np
import numpy
from misc import get_k_rank_mxn_matrix,random_binary_mask, validation, validation_objective
from als import als
import argparse
import random

def matrix_completion_app(partially_observed_M, args):
    mask = partially_observed_M!=0
    m,n = partially_observed_M.shape
    non_zero_count = numpy.sum(mask)
    zero_count = m*n - non_zero_count
    print('\nStarting Matrix Completion Application....'
          '\nInput matrix is [{0}x{1}]'
          '\nThe provided rank for factorization is {2}'
          '\nTotal {3} values in M are observed. {4} values are unknown.'.
          format(m,n,args.rank,non_zero_count,zero_count))
    final_U, final_V, initial_U, initial_V = als(partially_observed_M,mask,args)

    completed_matrix=numpy.dot(final_U,final_V.T)

    pre_train_loss = validation(partially_observed_M, mask, initial_U, initial_V)
    post_train_loss = validation(partially_observed_M, mask, final_U, final_V)

    print('Saving the processed matrices in out.npz')
    numpy.savez('out.npz',completed_matrix=completed_matrix,
                U=final_U,V=final_V)

    print('TRAINING RESULTS')
    print('Avg. Reconstruction loss on the observed matrix entries before training'
          ' (with random U and V) ={0}'.format(pre_train_loss))
    print('Avg. Reconstruction loss on the observed matrix entries after training'
          ' (with trained U and V) ={0}'.format(post_train_loss))

    return


def matrix_completion_demo(m,n,demo_rank,observed_fraction,args):
    fully_observed, true_rank = get_k_rank_mxn_matrix(m,n,demo_rank)
    print('\nSTARTING DEMO ..... \n')
    print('Running the matrix completion demo.'
          '\nWe will generate a random matrix of size M= [{0}x{1}], with {2} non-zero singular values (True SVD rank).'
          '\n{3} fraction of [{0}x{1}] entries will be masked out to simulate a partially observed matrix.'
          '\nWe will use ALS procedure to estimate matrices U and V, such that ||M - U*transpose(V)||^2 is minimum. '
          ' will reconstruct the fully observed matrix.'
          '\nU and V will have rank equal to the user input --rank parameter ={4}.'
          '\nWe will then compare the difference between the true unobserved entries of M and the predictions to validate our method.'
          .format(m,n,true_rank,1-observed_fraction,args.rank))

    mask = random_binary_mask(m,n,observed_fraction)
    partially_observed = fully_observed* mask
    final_U, final_V, initial_U, initial_V = als(partially_observed,mask,args)

    pre_train_loss = validation(partially_observed, mask,initial_U,initial_V)
    post_train_loss = validation(partially_observed, mask, final_U, final_V)

    # We validate our method on the matrix values that are hidden from the ALS procedure.
    unobserved_values = fully_observed * (1-mask)

    pre_validation_loss = validation(unobserved_values, (1-mask), initial_U,initial_V)
    post_validation_loss = validation(unobserved_values, (1-mask), final_U, final_V)

    pre_all_loss = validation(fully_observed, numpy.ones_like(mask),initial_U,initial_V)
    post_all_loss = validation(fully_observed, numpy.ones_like(mask), final_U, final_V)

    print('DEMO RESULTS')
    print('Avg. Reconstruction loss on the observed matrix entries before training'
          ' (with random U and V) ={0}'.format(pre_train_loss))
    print('Avg. Reconstruction loss on the observed matrix entries after training'
          ' (with trained U and V) ={0}'.format(post_train_loss))

    print('VALIDATION: Avg. Reconstruction loss on the unobserved matrix entries before training'
          ' (with random U and V) ={0}'.format(pre_validation_loss))
    print('VALIDATION: Avg. Reconstruction loss on the unobserved matrix entries after training'
          ' (with trained U and V) ={0}'.format(post_validation_loss))

    print('VALIDATION: Avg. Reconstruction loss on the all matrix entries before training'
          ' (with random U and V) ={0}'.format(pre_all_loss))
    print('VALIDATION: Avg. Reconstruction loss on the all matrix entries after training'
          ' (with trained U and V) ={0}'.format(post_all_loss))

    print('Saving the processed matrices in demo_out.npz')
    numpy.savez('demo_out.npz',matrix=fully_observed,mask=mask,
                U=final_U,V=final_V)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Matrix Completion Library using ALS. '
                                                 'Each LS step uses Gradient Descent. '
                                                 'Factorizes a matrix M of dimension [mxn] into U x transpose(V).')

    parser.add_argument('-d', '--run_demo', action='store_true',
                        help="Runs the demo")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="sets verbosity to True. False verbosity removes training messages.")
    parser.add_argument('-file','--filepath', type=str, default=None,
                        help='Filepath for the partially observed matrix M')
    parser.add_argument('-m', '--m', type=int, default=None,
                        help="m for demo matrix")
    parser.add_argument('-n', '--n', type=int, default=None,
                        help="n for demo matrix")
    parser.add_argument('-obs', '--observed_fraction', type=float, default=None,
                        help="The fraction of the demo matrix that should be observed."
                             "(1 - observed_fraction) is the fraction of matrix entries that are missing.")
    parser.add_argument('-k','--rank', type = int, default=5,
                        help='Rank of the required decomposition. Must be less than m and n')
    parser.add_argument('-ai','--max_ALS_steps', type=int, default=50,
                        help='Maximum number of ALS steps')
    parser.add_argument('-gi','--max_GD_steps', type=int, default=50,
                        help='Maximum number of GD steps for each least square optimization in ALS')

    args = parser.parse_args()

    if args.run_demo is False:
        if args.filepath is None:
            assert False, "Please either use the run_demo option or provide a matrix file for processing"
        else:
            partial_M = np.load(args.filepath)

        matrix_completion_app(partial_M,args)
    else:
        if args.m is None or args.n is None or args.observed_fraction is None:
            print('Either the m or n value or the demo observed_fraction is missing.'
                  ' Using random values to run the demo.')
            args.m = random.randint(100,1000)
            args.n = random.randint(50,100)
            args.observed_fraction = random.random()
        demo_rank= random.randint(5,min(10,args.m, args.n))

        matrix_completion_demo(args.m,args.n,demo_rank,args.observed_fraction,args)