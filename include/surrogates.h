/****************************************************************************************/
/* surrogates.h                                                                         */
/*                                                                                      */
/* functions for shuffling datasets.                                                    */
/*                                                                                      */
/* Created by Nicolas Garnier on 2023/01/30.                                            */
/* Copyright 2010-2023 ENS-Lyon - CNRS. All rights reserved.                            */
/*                                                                                      */
/* 2023-01-30 : forked from "sampling.h"                                                */
/* 2023-02-27 : added Fourier method (with fftw)                                        */
/****************************************************************************************/

#ifndef SURROGATES_H
#define SURROGATES_H

// #include <gsl/gsl_permutation.h>

// private:
int  init_surrogate_FFTW(int npts);
void free_surrogate_FFTW(void);
void init_surrogates_rng(void);

// public:
void    surrogate_uFt   (double *x, int npts, int mx);              // in-place!
void    surrogate_wFt   (double *x, int npts, int mx);              // in-place!
void    surrogate_aaFt  (double *x, int npts, int mx);              // in-place!
void  surrogate_improved(double *x, int npts, int mx, int N_steps); // in-place!
void    Gaussianize     (double *x, int npts, int mx);              // in-place!
void    shuffle_data    (double *x, int npts, int mx);              // in-place!
double *create_surrogate(double *x, int npts, int mx);              // returns new pointer

#endif
