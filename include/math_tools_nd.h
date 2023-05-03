/*
 *  math-tools.h
 *  
 *
 *  Created by Nicolas Garnier on 13/08/10.
 *  Copyright 2010-2021 ENS-Lyon CNRS. All rights reserved.
 *
 * 2021-01-19: split from "math_tools.c" for clarity
 */

#include <math.h>


double normalize    (double *x, int nx); /* normalize with respect to its variance (returned value) */

int compute_covariance_matrix(double *x, int m, int npts, double *Sigma);
double determinant_covariance(double *x, int m, int npts);
double normalize_nd          (double *x, int m, int npts);

