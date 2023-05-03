/*
 *  entropy_Gaussian.h
 *  
 *
 *  Created by Nicolas Garnier on 2022/10/10.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  This set of functions operates on continuous data !!!
 *
 *  2022-10-10 : forked from "entropy_ann_N.h"
 */



/********************************************************************************************************************/
/* function to compute Shannon entropy */
int compute_entropy_Gaussian  	        (double *x, int nx, int m, int p, int tau, int tau_Theiler, 
                                    int N_eff, int N_realizations, double *S); /* wrapper, to be used */
                                    
/* function to compute relative entropy */
//int compute_relative_entropy_Gaussian   (double *x, int nx, double *y, int ny, int mx, int my, int px, int py, int tau, int tau_Theiler, 
//                                    int N_eff, int N_realizations, double *H); /* wrapper, to be used */

// for the functions below, 2 parameters are 2 returned values : I1 and I2, corresponding to algorithm 1 and 2

// function to compute mutual information:
int compute_mutual_information_Gaussian (double *x, double *y, int nx, int mx, int my, int px, int py, int tau, int tau_Theiler, 
                                    int N_eff, int N_realizations, double *I1);
// functions to compute partial mutual information:
int compute_partial_MI_Gaussian         (double *x, double *y, double *z, int nx, int *dim, /*int dim_z, int m, int p, int q, */ 
                                        int tau, int tau_Theiler, int N_eff, int N_realizations, double *I1);

