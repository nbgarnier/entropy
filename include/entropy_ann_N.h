/*
 *  entropy_ann_N.h
 *  
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2019 ENS-Lyon - CNRS. All rights reserved.
 *
 *  The method used in these functions is from Grassberger (2004), 
 *  using nearest neighbor statistics and ANN library 
 *  http://www.cs.umd.edu/~mount/ANN/
 *
 *  This set of functions operates on continuous data !!!
 *
 *  2019-01-25 : renamed "compute_mutual_information_nd_ann" as "compute_mutual_information_ann"
 *  2022-04-15 : forked from "entropy_ann.h"
 */

/********************************************************************************************************************/
/* function to compute Shannon entropy */
int compute_entropy_ann_N 	        (double *x, int nx, int m, int p, int tau, int tau_Theiler, 
                                    int N_eff, int N_realizations, int k, double *S); /* wrapper, to be used */

/* functions to compute relative entropy */
int compute_relative_entropy_ann_N  (double *x, int nx, double *y, int ny, int mx, int my, int px, int py, int tau, int tau_Theiler, 
                                    int N_eff, int N_realizations, int k, double *H); /* wrapper, to be used */


// for the functions below, 2 parameters are 2 returned values : I1 and I2, corresponding to algorithm 1 and 2

// function to compute mutual information:
int compute_mutual_information_ann_N(double *x, double *y, int nx, int mx, int my, int px, int py, int tau, int tau_Theiler, 
                                    int N_eff, int N_realizations, int k, double *I1, double *I2);

// functions to compute partial mutual information:
int compute_partial_MI_ann_N        (double *x, double *y, double *z, int nx, int *dim, /*int dim_z, int m, int p, int q, */ 
                                    int tau, int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);

