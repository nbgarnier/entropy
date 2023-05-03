/*
 *  entropy_ann_single_MI.h
 *  
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *  New functions to be imported in entropy_ann.h/c
 *
 *  2019-01-30 : file "entropy_ann_new.h" created, forked from "entropy_ann.h"
 *  2021-12-08 : file created, forked from "entropy_ann_new.h"
 */

/* low-level function to compute mutual information	*/
int compute_mutual_information_direct_ann(
    double *x, int nx, int m, int p,        int k, double *I1, double *I2);
    /* 2 datasets of dimension p and k */
