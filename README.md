# ALS_Matrix_Completion


This python library uses alternating Least Squares minimization for matrix completion. The minimization procedure uses Gradient Descent with decaying learning rate. Gradient Descent is implimented using autograd package (numpy).


## Library Usage 

> python main.py --help 
```
usage: main.py [-h] [-d] [-v] [-file FILEPATH] [-m M] [-n N] [-obs OBSERVED_FRACTION] [-k RANK] [-ai MAX_ALS_STEPS] [-gi MAX_GD_STEPS]

Matrix Completion Library using ALS. Each LS step uses Gradient Descent. Factorizes a matrix M of dimension [mxn] into U x transpose(V).

optional arguments:
  -h, --help            show this help message and exit
  -d, --run_demo        Runs the demo
  -v, --verbose         sets verbosity to True. False verbosity removes training messages.
  -file FILEPATH, --filepath FILEPATH
                        Filepath for the partially observed matrix M
  -m M, --m M           m for demo matrix
  -n N, --n N           n for demo matrix
  -obs OBSERVED_FRACTION, --observed_fraction OBSERVED_FRACTION
                        The fraction of the demo matrix that should be observed.(1 - observed_fraction) is the fraction of matrix entries that are missing.
  -k RANK, --rank RANK  Rank of the required decomposition. Must be less than m and n
  -ai MAX_ALS_STEPS, --max_ALS_steps MAX_ALS_STEPS
                        Maximum number of ALS steps
  -gi MAX_GD_STEPS, --max_GD_steps MAX_GD_STEPS
                        Maximum number of GD steps for each least square optimization in ALS
```
## Sample Demo Run

A demo run can be run by using 

> python main.py --run_demo 

Sample Result : 
---
  ``` Either the m or n value or the demo observed_fraction is missing. Using random values to run the demo.

  STARTING DEMO ..... 

  Running the matrix completion demo.
  We will generate a random matrix of size M= [135x60], with 7 non-zero singular values (True SVD rank).
  0.9849628923318247 fraction of [135x60] entries will be masked out to simulate a partially observed matrix.
  We will use ALS procedure to estimate matrices U and V, such that ||M - U*transpose(V)||^2 is minimum.  will reconstruct the fully observed matrix.
  U and V will have rank equal to the user input --rank parameter =5.
  We will then compare the difference between the true unobserved entries of M and the predictions to validate our method.

  STARTING TRAINING ..... 

	 TRAINING: ALS STEP: 0 / 50
	 TRAINING: ALS STEP: 1 / 50
	 TRAINING: ALS STEP: 2 / 50
	 TRAINING: ALS STEP: 3 / 50
	 TRAINING: ALS STEP: 4 / 50
	 TRAINING: ALS STEP: 5 / 50
	 TRAINING: ALS STEP: 6 / 50
	 TRAINING: ALS STEP: 7 / 50
	 TRAINING: ALS STEP: 8 / 50
	 TRAINING: ALS STEP: 9 / 50
	 TRAINING: ALS STEP: 10 / 50
	 TRAINING: ALS STEP: 11 / 50
	 TRAINING: ALS STEP: 12 / 50
	 TRAINING: ALS STEP: 13 / 50
	 TRAINING: ALS STEP: 14 / 50
	 TRAINING: ALS STEP: 15 / 50
	 TRAINING: ALS STEP: 16 / 50
	 TRAINING: ALS STEP: 17 / 50
	 TRAINING: ALS STEP: 18 / 50
	 TRAINING: ALS STEP: 19 / 50
	 TRAINING: ALS STEP: 20 / 50
	 TRAINING: ALS STEP: 21 / 50
	 TRAINING: ALS STEP: 22 / 50
	 TRAINING: ALS STEP: 23 / 50
	 TRAINING: ALS STEP: 24 / 50
	 TRAINING: ALS STEP: 25 / 50
	 TRAINING: ALS STEP: 26 / 50
	 TRAINING: ALS STEP: 27 / 50
	 TRAINING: ALS STEP: 28 / 50
	 TRAINING: ALS STEP: 29 / 50
	 TRAINING: ALS STEP: 30 / 50
	 TRAINING: ALS STEP: 31 / 50
	 TRAINING: ALS STEP: 32 / 50
	 TRAINING: ALS STEP: 33 / 50
	 TRAINING: ALS STEP: 34 / 50
	 TRAINING: ALS STEP: 35 / 50
	 TRAINING: ALS STEP: 36 / 50
	 TRAINING: ALS STEP: 37 / 50
	 TRAINING: ALS STEP: 38 / 50
	 TRAINING: ALS STEP: 39 / 50
	 TRAINING: ALS STEP: 40 / 50
	 TRAINING: ALS STEP: 41 / 50
	 TRAINING: ALS STEP: 42 / 50
	 TRAINING: ALS STEP: 43 / 50
	 TRAINING: ALS STEP: 44 / 50
	 TRAINING: ALS STEP: 45 / 50
	 TRAINING: ALS STEP: 46 / 50
	 TRAINING: ALS STEP: 47 / 50
	 TRAINING: ALS STEP: 48 / 50
	 TRAINING: ALS STEP: 49 / 50
  DEMO RESULTS
  Avg. Reconstruction loss on the observed matrix entries before training (with random U and V) =0.8697641297394634
  Avg. Reconstruction loss on the observed matrix entries after training (with trained U and V) =0.02964950081119748
  VALIDATION: Avg. Reconstruction loss on the unobserved matrix entries before training (with random U and V) =0.7385105555540147
  VALIDATION: Avg. Reconstruction loss on the unobserved matrix entries after training (with trained U and V) =1.0171558288636235
  VALIDATION: Avg. Reconstruction loss on the all matrix entries before training (with random U and V) =0.7404712570943158
  VALIDATION: Avg. Reconstruction loss on the all matrix entries after training (with trained U and V) =1.0024041911235813
  Saving the processed matrices in demo_out.npz ```
