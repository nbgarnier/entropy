/*
 *  entropy_others.h
 *  
 *
 *  Created by Nicolas Garnier on 2013/04/19.
 *  Copyright 2013 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2013-04-19 : fork from entropy_symb
 * 
 */

#ifndef ENTROPY_OTHERS_H
#define ENTROPY_OTHERS_H

#define KERNEL_BRICKWALL  0x00
#define KERNEL_GAUSSIAN   0x01

// these first 2 functions are old version (05/2013):
double compute_ApEn_old        (double* data, int m, double r, int npts);
double compute_SampEn_old      (double* data, int m, double r, int npts);

// these are new functions (03/2014):
double prepare_kernel_brickwall_embed(double* data, int npts, int m, int stride, double r);
double prepare_kernel_Gaussian_embed(double* data, int npts, int m, int stride, double r);
void   free_complexity_kernel  (void);
int    compute_complexity      (double* data, int npts, int m, int stride, double r, 
                                    int kernel_type, double *ApEn, double *SampEn);

// and these are even newer functions (2023-12-06):
void    *alloc_complexity_kernel(int npts, int m);
void    reset_array_d           (double *x, int n);
double  apply_kernel_brickwall_nd(double* data, int npts, int m, double r);
double  apply_kernel_Gaussian_nd(double* data, int npts, int m, double r);
int     compute_complexity_mask (double* data, char *mask, int npts, int m, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, 
                            double r, int kernel_type, double *ApEn, double *SampEn);

#endif
