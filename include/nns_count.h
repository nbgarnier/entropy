/*
 *  nns_count.h
 *  
 *
 *  Created by Nicolas Garnier on 2019-01-23 (fork from 2010)
 *  Copyright 2010-2019 ENS-Lyon CNRS. All rights reserved.
 *
 *  These functions are applications of the theory of Grassberger (2004),
 *  using nearest neighbor statistics 
 *
 */

/* functions to count nearest neighbors of a point y0 in a ball of radius epsilon */
int count_nearest_neighbors_1d_algo1 (double *y, int nx, int i, double y0, double epsilon); // for algo 1 (<eps)
int count_nearest_neighbors_1d_algo2 (double *y, int nx, int i, double y0, double epsilon); // for algo 2 (<=eps)
int count_nearest_neighbors_nd_algo1 (double *y, int nx, int n, int i, int *ind_inv, double *y0, double  epsilon); // for algo 1 (<eps)
int count_nearest_neighbors_nd_algo2 (double *y, int nx, int n, int i, int *ind_inv, double *y0, double  epsilon); // for algo 2 (<=eps)
int count_nearest_neighbors_nd_algo2n(double *y, int nx, int n, int i, int *ind_inv, double *y0, double *epsilon); // for algo 2 (<=eps)
int count_nearest_neighbors_embed_nd_algo1(double **A, double *x_sorted, int nx, int n, int i, int *ind_inv, double *y0, double epsilon);
