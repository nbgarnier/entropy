/*
 *  entropy_ann_single_RE.h
 *  
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *  Functions to be imported in entropy_ann.h/c
 *
 *  2021-12-14 : file created, forked from "entropy_ann.h"
 */

/* low-level function to compute relative entropy */
double compute_relative_entropy_2xnd_ann(double *x, int nx, double *y, int ny, int n, int k);
