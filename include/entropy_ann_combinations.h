/*
 *  entropy_ann_combinations.h
 *  
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  The method used in these functions is from Grassberger (2004), 
 *  using nearest neighbor statistics and ANN library 
 *  http://www.cs.umd.edu/~mount/ANN/
 *
 *  This set of functions operates on continuous data !!!
 *
 *  2019-01-25 : renamed "compute_mutual_information_nd_ann" as "compute_mutual_information_ann"
 *  2021-12-21 : new file "entropy_ann_combinations.h" split from "entropy_ann.h"
 *  2022-05-26 : new samplings for all functions
 */


// Shannon entropy of increments (using entropy) (returning 1 value):
int compute_entropy_increments_ann  (double *x, int nx, int m, int p, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, int incr_type, double *S);

// entropy rate (using entropy and/or MI)*/
int compute_entropy_rate_ann        (double *x, int nx, int m, int p, int stride,
                            int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *H);

// for the functions below, there are 2 returned values : I1 and I2, corresponding to algorithm 1 and 2

// Delta / regularity/non-stationarity index (using MI):
int compute_regularity_index_ann    (double *x, int npts, int mx, int px, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);

// transfer entropy (using PMI):
int compute_transfer_entropy_ann    (double *x, double *y, int nx, int mx, int my, int px, int py, int stride, int lag, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *T1, double *T2);

// directed information (using PMI):
int compute_directed_information_ann(double *x, double *y, int nx, int mx, int my, int N, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);


