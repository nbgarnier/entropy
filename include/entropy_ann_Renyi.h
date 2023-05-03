/*
 *  entropy_ann_Renyi.h
 *
 *  Created by Nicolas Garnier on 05/10/2014.
 *  Copyright 2014 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2014-10-05 : new function "compute_Renyi_nd_ann"
 *
 */
#ifndef __entropy_ANN_Renyi__
#define __entropy_ANN_Renyi__


double compute_Renyi_nd_ann(double *x, int nx, int n, double q, int k);
int    compute_Renyi_ann   (double *x, int nx, int m, int p, int tau, double q, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S);

#endif /* defined(__entropy_ANN_Renyi__) */
