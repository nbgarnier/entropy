/*
 *  entropy_ann_mask.h
 *  
 *
 *  Created by Nicolas Garnier on 2013/06/20.
 *  Copyright 2013 ENS-Lyon - CNRS. All rights reserved.
 *
 *  Masking facilities (epochs)
 *
 */


/* functions to compute Shanon or Renyi entropy */
int compute_entropy_ann_mask         (double *x, char *mask, int npts, int m, int p, int stride,           
                            int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S);
int compute_Renyi_ann_mask           (double *x, char *mask, int npts, int m, int p, int stride, double q, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S);

/* functions to compute entropy rate */
int compute_entropy_rate_ann_mask      (double *x, char *mask, int npts, int m, int p, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S);

/* functions to compute mutual information	*/
/* 2 parameters are 2 returned values : I1 and I2, corresponding to algorithm 1 and 2   */
int compute_mutual_information_ann_mask(double *x, double *y, char *mask, int npts, int mx, int my, int px, int py, int stride,
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);

/* functions to compute transfer entropy	*/
int compute_transfer_entropy_ann_mask  (double *x, double *y, char *mask, int npts, int mx, int my, int px, int py, int stride, int lag, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *T1, double *T2); // uses PMI

/* functions to compute partial transfer entropy */
int compute_partial_TE_ann_mask        (double *x, double *y, double *z, char *mask, int nx, int *dim, int stride, int lag, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);
/* functions to compute partial mutual information */
int compute_partial_MI_ann_mask        (double *x, double *y, double *z, char *mask, int nx, int *dim, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);

/* function for Directed Information : */
int compute_directed_information_ann_mask(double *x, double *y, char *mask, int npts, int mx, int my, int N, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2);

