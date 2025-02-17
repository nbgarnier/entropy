/*
 *  entropy_Gaussian_single.h
 *  
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  Functions to be imported in entropy_Gaussian.h/c
 *
 */

/* low-level function to compute entropy */
double compute_entropy_nd_Gaussian	               (double *x, int nx, int n);	/* internal function */

/* low-level function to compute relative entropy */
double compute_relative_entropy_nd_Gaussian(double *x, int npts_x, double *y, int npts_y, int n, int method);

/* low-level function to compute mutual information	*/
double compute_mutual_information_direct_Gaussian  (double *x, int nx, int m, int p);
    /* 2 datasets of dimension p and k */

/* low-level function to compute partial mutual information	*/
double compute_partial_MI_engine_Gaussian          (double *x, int npts, int mx, int my, int mz);
    /* 3 datasets of dimension mx, my and mz */

