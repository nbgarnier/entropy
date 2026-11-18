/****************************************************************************************/
/* samplings_random.c                                                                   */
/*                                                                                      */
/* low level functions dealing with random numbers generation for sampling datasets.    */
/*                                                                                      */
/* Created by Nicolas Garnier on 2026/11/10 (fork from samplings.c)                     */
/* Copyright 2010-2026 ENS-Lyon - CNRS. All rights reserved.                            */
/*                                                                                      */
/* 2026-11-10 : new file with 3 functions from "samplings.c"                            */
/****************************************************************************************/

#include <gsl/gsl_rng.h>                // for random permutations
#include <gsl/gsl_randist.h>

#include "samplings_random.h"           // for these functions

static int samplings_pts_rng_state=0;       // to keep track of the rng state for random sampling of pts within a realization
static int samplings_real_rng_state=0;      // to keep track of the rng state for random sampling of realizations
// 2026-11-10 seeds for the two rng:
// note: seed are here signed int, to keep the posssibillity of (-1) (not touching the seed, legacy behavior)
long int samplings_pts_rng_seed=-1;         // -1 is the legacy default (numbers are always different, no seed used)
long int samplings_real_rng_seed=-1; 

static gsl_rng *samplings_pts_rng;          // random nb generator for sampling of pts within a realization
static gsl_rng *samplings_real_rng;         // random nb generator for sampling of realizations



/****************************************************************************************/
/* set the seed (if >=0 value provided) for the random nb generator used for sampling   */
/*  or leave it untouched (if <0 value provided) so that numbers are always different,  */
/*  which is the expected (and legacy) behavior)                                        */
/*                                                                                      */
/* 2026-11-10 : we are lazy here and use the same seed for the two rng...               */
/*              this may be changed later if needed                                     */
/****************************************************************************************/
void seed_samplings_rng(long int seed)
{   if (seed>=0) 
    {   samplings_pts_rng_seed  = (unsigned long int)seed;
        samplings_real_rng_seed = (unsigned long int)seed;    
    }
    else
    {   samplings_pts_rng_seed  = -1;
        samplings_real_rng_seed = -1;    
    }
}


/****************************************************************************************/
/* init the random number generators for random sampling                                */
/****************************************************************************************/
void init_samplings_rng(void)
{   const gsl_rng_type *T;

    // for sampling of points within a given realisation:
    if (samplings_pts_rng_state==0)
    {   gsl_rng_env_setup();
        T = gsl_rng_default;
        samplings_pts_rng = gsl_rng_alloc(T);
        samplings_pts_rng_state++;
    }

    // for random realisations:
    if (samplings_real_rng_state==0)
    {   gsl_rng_env_setup();
        T = gsl_rng_default;
        samplings_real_rng = gsl_rng_alloc(T);
        samplings_real_rng_state++;
    }
   
    // at this stage, the seeds used are gsl_rng_default_seed (0 by default, but can be set by environment variable GSL_RNG_SEED)

    // 2026-11-10: initialize or set the seed (only if required):
    if (samplings_pts_rng_seed>=0)  gsl_rng_set(samplings_pts_rng,  (unsigned)samplings_pts_rng_seed);
    if (samplings_real_rng_seed>=0) gsl_rng_set(samplings_real_rng, (unsigned)samplings_real_rng_seed);
}


/****************************************************************************************/
/* init the random number generators for random sampling and allocate permutations      */
/*                                                                                      */
/****************************************************************************************/
void init_samplings(gsl_permutation **perm_real, int N_real, gsl_permutation **perm_pts, int N_pts)
{   
    init_samplings_rng();

    if (N_real>0)   *perm_real = create_unity_perm(N_real); 
    else            *perm_real = NULL;
    if (N_pts>0)    *perm_pts  = create_unity_perm(N_pts);     // for random sampling
    else            *perm_pts  = NULL;

#ifdef DEBUG
    printf("[init_samplings] called with N_real=%d and N_pts=%d, OK\n", N_real, N_pts);
    if (*perm_real!=NULL)   printf("[init_samplings] realisations permutation pointer OK\n");
    else                    printf("[init_samplings] realisations permutation pointer NULL\n");
    if (*perm_pts!=NULL)    printf("[init_samplings] pts permutation pointer OK\n");
    else                    printf("[init_samplings] pts permutation pointer NULL\n");
#endif

    return;
}


/****************************************************************************************/
/* creates and returns a permutation of N elements                                      */
/*                                                                                      */
/* note that the permutation is allocated on the fly by this function,                  */
/* so it should be properly de-allocated later after use                                */
/****************************************************************************************/
gsl_permutation *create_unity_perm(size_t N)
{   gsl_permutation *perm;
    
//    init_samplings_rng(); // 2026-11-10: removed from here: don't forget to do it outside!

    perm = gsl_permutation_alloc(N);
    gsl_permutation_init(perm);     // set the permutation to identity

    return(perm);
}


/****************************************************************************************/
/* creates a new random permutation of N elements, in-place                             */
/*                                                                                      */
/* note that the permutation should be already allocated                                */
/* one function for the "realization" and one function for the "points"                 */
/* (each function uses its own rng)                                                     */
/****************************************************************************************/
void shuffle_samplings_real_perm(gsl_permutation *perm)
{   if (samplings_real_rng_state>0)
        gsl_ran_shuffle(samplings_real_rng, perm->data, perm->size, sizeof(size_t));

    return;
}

void shuffle_samplings_pts_perm(gsl_permutation *perm)
{   if (samplings_pts_rng_state>0)
        gsl_ran_shuffle(samplings_pts_rng, perm->data, perm->size, sizeof(size_t));

    return;
}


/****************************************************************************************/
/* deallocates all memory used by sampling permutations (realisations and points)       */
/*                                                                                      */
/* 2026-11-10 : new                                                                     */
/****************************************************************************************/
void free_samplings(gsl_permutation *perm_real, gsl_permutation *perm_pts)
{   //if (realizations_rng_state>0) realizations_rng_state--;
/*    if (realizations_rng_state>0)
    {   gsl_rng_free(realizations_rng);
        realizations_rng_state = 0;
    }
*/  // 2022-05-24: I now postpone the cleaning of the rng to the code exiting
    gsl_permutation_free(perm_real);
    gsl_permutation_free(perm_pts);
    return;
}

