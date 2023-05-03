/*
 *  entropy_ann_single_PMI.h
 *  
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *  New functions to be imported in entropy_ann.h/c
 *
 *  2019-01-30 : file "entropy_ann_new.h" created, forked from "entropy_ann.h"
 *  2021-12-08 : file "entropy_ann_single_MI.h" created, forked from "entropy_ann_new.h"
 *  2021-12-14 : file created, forked from "entropy_ann_single_MI.h"
 */

/* low-level function to compute partial mutual information	*/
int compute_partial_MI_engine_ann(
    double *x, int npts, int mx, int my, int mz, int k, double *I1, double *I2);
    /* 3 datasets of dimension mx, my and mz */
