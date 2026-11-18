/****************************************************************************************/
/* samplings_random.h                                                                   */
/*                                                                                      */
/* low level functions dealing with random numbers generation for sampling datasets.    */
/*                                                                                      */
/* Created by Nicolas Garnier on 2026/11/10 (fork from samplings.h)                     */
/* Copyright 2010-2026 ENS-Lyon - CNRS. All rights reserved.                            */
/*                                                                                      */
/* 2026-11-10 : new file, with some declarations from samplings.h                       */
/****************************************************************************************/
#ifndef SAMPLINGS_RANDOM_H
#define SAMPLINGS_RANDOM_H

#include <gsl/gsl_permutation.h>

extern long int samplings_pts_rng_seed;  // -1 is the legacy default (numbers are always different, no seed used)
extern long int samplings_real_rng_seed; 

/****************************************************************************************/
// functions to handle enhanced sampling/shuffling parameters:
void seed_samplings_rng             (long int seed);
// the main ones to use (they do all the init jobs, and free jobs):
void init_samplings                 (gsl_permutation **perm_real, int N_real, gsl_permutation **perm_pts, int N_pts); 
void free_samplings                 (gsl_permutation *perm_real, gsl_permutation *perm_pts);

// void shuffle_perm                   (gsl_permutation *perm); // deprecated
void shuffle_samplings_real_perm    (gsl_permutation *perm);
void shuffle_samplings_pts_perm     (gsl_permutation *perm);

/****************************************************************************************/
// internal functions:
// void init_samplings_rng             (void);
gsl_permutation *create_unity_perm  (size_t N);

#endif
