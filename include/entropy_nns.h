/*
 *  entropy_nns.h
 *  
 *
 *  Created by Nicolas Garnier on 20/08/10.
 *  Copyright 2010 ENS-Lyon CNRS. All rights reserved.
 *
 *  The method used in these functions is from Grassberger (2004), 
 *  using nearest neighbor statistics 
 *
 *  This set of functions operates on continuous data !!!
 *
 */

#define USE_THEILER // this define will propagate in entropy_nns.c and entrop_ann.c
        // when USE_THEILER is defined, the Theiler correction is used in the computation of MI, TE, and PMI

/* variables used "globally" but by internal functions only		*/
/* ie, there should be no point to declare them here globally	*/
/* extern  int	   stride;
extern	double *yi;
extern	double *yl;
extern	double *epsilon_l;
extern	double *epsilon_j;
*/

/* functions to allocate memory for internal functions			*/
int	init_nns(int nx, int n_max);
void	free_nns(void);

/* functions to compute Shanon entropy */
double compute_entropy_1d_nns       (double *x, int nx, int k);			/* conform to Grassberger / KL			*/
double compute_entropy_nd_nns		(double *x, int nx, int n, int k);	/* using "search_nearest_neighbors()" */
int    compute_entropy_nns			(double *x, int nx, int m, int p, int stride, int k, double *S); /* wrapper, to be used */
int    compute_entropy_embed_nns    (double *x, int nx, int n,        int stride, int k, double *H); // a little slower than the one above, but much more memory efficient

/* functions to compute entropy rate */
int    compute_entropy_rate_nns     (double *x, int nx, int n,        int stride, int k, double *H);  

/* functions to search for nearest neighbors of a point	*/
int search_nearest_neighbors         (double *x,                    int nx, int n, int i, int k, int *li, double *epsilon_z);
int search_nearest_neighbors_1_stride(double **x, double *x_sorted, int nx, int n, int i, int k, int *li, double *epsilon_z);

/* functions to count nearest neighbors of a point y0 in a ball of radius epsilon */
int count_nearest_neighbors_1d_algo1 (double *y, int nx, int i, double y0, double epsilon); // for algo 1 (<eps)
int count_nearest_neighbors_1d_algo2 (double *y, int nx, int i, double y0, double epsilon); // for algo 2 (<=eps)
int count_nearest_neighbors_nd_algo1 (double *y, int nx, int n, int i, int *ind_inv, double *y0, double  epsilon); // for algo 1 (<eps)
int count_nearest_neighbors_nd_algo2 (double *y, int nx, int n, int i, int *ind_inv, double *y0, double  epsilon); // for algo 2 (<=eps)
int count_nearest_neighbors_nd_algo2n(double *y, int nx, int n, int i, int *ind_inv, double *y0, double *epsilon); // for algo 2 (<=eps)
int count_nearest_neighbors_embed_nd_algo1(double **A, double *x_sorted, int nx, int n, int i, int *ind_inv, double *y0, double epsilon);

/* functions to compute mutual information	*/
/* 2 parameters are 2 returned values : h1 and h2, corresponding to algorithm 1 and 2   */
int compute_mutual_information_nx1d_nns(double *x, int nx,    int n,     int k, double *I1, double *I2); /* n datasets of 1d */ /* ! this is the function to use ! */
int compute_mutual_information_2xnd_nns(double *x, int nx, int m, int p, int k, double *I1, double *I2); /* 2 datasets of dimension p and k */
int	compute_mutual_information_2x1d_stride_nns(double *x, double *y, int nx, int m, int p, int stride, int k, double *I1, double *I2); /* 2 datasets of dimension 1, with stride in time */
int compute_mutual_information_nns            (double *x, double *y, int nx, int m, int p, int stride, int k, double *I1, double *I2); /* same as above */
int compute_mutual_information_2wip_nd_nns    (double *x, double *y, int nx, int m, int p, int k, double *I1, double *I2);

/* functions to compute transfer entropy	*/
int compute_transfer_entropy_from_MI_2x1d_nns(double *x, double *y, int nx, int k, double *T1, double *T2); // OK
int compute_transfer_entropy_from_MI_2xnd_nns(double *x, double *y, int nx, int m, int p, int stride, int k, double *T1, double *T2);
int compute_transfer_entropy_from_MI_nns     (double *x, double *y, int nx, int m, int p, int stride, int lag, int k, double *T1, double *T2); // wrapper
int compute_transfer_entropy_direct_nns      (double *x, int nx, int m, int p, int k, double *I1, double *I2); // 2011-09-21, the good one !!!
int compute_transfer_entropy_nns             (double *x, double *y, int nx, int m, int p, int stride, int lag, int k, double *T1, double *T2); // wrapper for the good one

/* functions to compute partial mutual information */
int compute_partial_MI_direct_nns             (double *x,		                int nx, int m, int p, int q,             int k, double *I1, double *I2);
int compute_partial_MI_nns                    (double *x, double *y, double *z, int nx, int m, int p, int q, int stride, int k, double *I1, double *I2);

/* old functions : */
#include "entropy_nns_old.h"


