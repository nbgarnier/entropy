/*
 *  entropy_ann_2d.h
 *  
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  New functions to be imported in entropy_ann.h/c
 *
 *  2020-07-19: file created, forked from entropy_ann.h
 *  2020-09-01: using struct for parameters
 *  2022-01-12: multithread function for entropy (embedding & various increments)
 */



/* high-level function (wrapper) to compute entropy of images, with embedding on the fly */
//int compute_entropy_ann_2d(double *x, dimension_parameters dim, embedding_parameters embed, int k, double *S);
int compute_entropy_ann_2d(double *x, int nx, int ny, int d, int p, int stride_x, int stride_y, 
                            int Theiler_x, int Theiler_y, int N_eff, int N_realizations, int k, int method, double *S);
