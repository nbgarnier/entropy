/****************************************************************************************/
/* surrogates.c                                                                         */
/*                                                                                      */
/* functions for shuffling datasets.                                                    */
/*                                                                                      */
/* Created by Nicolas Garnier on 2023/01/30.                                            */
/* Copyright 2010-2023 ENS-Lyon - CNRS. All rights reserved.                            */
/*                                                                                      */
/* 2023-01-30 : forked from "sampling.c"                                                */
/* 2023-02-27 : added Fourier method (with fftw) corresponding to                       */
/*              Theiler et al Physica D 58 pp77-94 (1992)                               */
/* 2023-03-01 : Jupyter notebook causality_couples_2023-03-01_test_surrogates.ipynb     */
/****************************************************************************************/
#include <stddef.h>                     // for size_t
#include <string.h>                     // for memcpy
#include <math.h>                       // for trunc (to replace by %)
#ifndef M_PI
#define M_PI 3.14159265358979323846     // 2023-03-02 for phycalc2
#endif 

#include <gsl/gsl_rng.h>                // for random permutations
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_statistics_double.h>  // for mean
#include <fftw3.h>

#include "verbosity.h"
#include "math_tools.h"

#include "surrogates.h"


// following are used by basic shuffling method, AND by FT surrogates methods:
static int surrogates_rng_state=0;  // to keep track of the rng state
static gsl_rng *surrogates_rng;

// following is used by aaFt surrogate method:    
static int *surrogates_indices; 
static int *surrogates_ind_inv; 

// following are used by FT surrogates methods:
static int surrogates_FFT_state=0;  // to keep track of the FFT state
static fftw_complex *data_Ft;
static double       *phase_shift;
static double       *data_in; 
static fftw_plan plan_surrogate_direct;
static fftw_plan plan_surrogate_inverse;


/***********************************************************************/
/* initialization, i.e., allocation of memory				           */
/***********************************************************************/
int init_surrogate_FFTW(int npts)
{		
	if (surrogates_FFT_state!=npts) free_surrogate_FFTW();  // can the old plan be re-used?
	 
	// to store the Fourier transform:
	data_in     = (double*)      fftw_malloc( npts     *sizeof(double));   
    data_Ft     = (fftw_complex*)fftw_malloc((npts/2+1)*sizeof(fftw_complex));  // compact output is of size n/2+1 
    phase_shift = (double*)      fftw_malloc((npts/2+1)*sizeof(double)); 
	
    // fftw3 version of the plans: ( http://fftw.org/fftw3_doc/One_002dDimensional-DFTs-of-Real-Data.html
	plan_surrogate_direct  = fftw_plan_dft_r2c_1d(npts, data_in, data_Ft, FFTW_ESTIMATE);
	plan_surrogate_inverse = fftw_plan_dft_c2r_1d(npts, data_Ft, data_in, FFTW_ESTIMATE);

    surrogates_FFT_state = npts;
	return(npts);
}

/***********************************************************************/
/* free the memory allocated by GSCN_init				               */
/***********************************************************************/
void free_surrogate_FFTW(void)
{	
    if (surrogates_FFT_state>0)
    {   fftw_destroy_plan(plan_surrogate_direct);
	    fftw_destroy_plan(plan_surrogate_inverse);

        fftw_free(data_in);
	    fftw_free(data_Ft);
	    fftw_free(phase_shift);
	    
	    surrogates_FFT_state=0;
	}
}


/****************************************************************************************/
/* creates a surrogate of a (possibly multi-dimensional) dataset of npts elements       */
/* in-place                                                                             */
/* "unwindowed Fourier transform" from Theiler et al Physica D 58 (1992) 2.4.1          */
/* see also Prichard & Theiler PRL 73 (7) (1994) for multi-dimensional signals          */
/*                                                                                      */
/* 2023-02-27 : new function                                                            */
/* 2023-02-28 : tested OK.                                                              */
/****************************************************************************************/
void surrogate_uFt(double *x, int npts, int mx)
{   double u,v, mean, mtmp;
    int d, i;
    
    (void)init_surrogate_FFTW(npts);
    init_surrogates_rng();
    
    // create random phase shifts:
    for (i=0; i<(npts/2+1); i++) phase_shift[i]=gsl_rng_uniform(surrogates_rng);
    
    for (d=0; d<mx; d++)
    {   memcpy(data_in, x+d*npts, npts*sizeof(double));
        mean = gsl_stats_mean(data_in, 1, (size_t)npts);
        fftw_execute(plan_surrogate_direct);
        
        // randomize the phase by applying random phase shifts (same for all dimensions to conserve cross-correlations):
        for (i=0; i<(npts/2+1); i++)
	    {	u = data_Ft[i][0];
	        v = data_Ft[i][1];
	        data_Ft[i][0] = (u*cos(phase_shift[i]) - v*sin(phase_shift[i]))/npts;
		    data_Ft[i][1] = (u*sin(phase_shift[i]) + v*cos(phase_shift[i]))/npts;
	    }
	    // note that /npts is for the FFTw normalization, and it is speed-efficient to do it at this stage
	    
        fftw_execute(plan_surrogate_inverse);
        
        // 2023-03-01: added the following:
        mtmp = gsl_stats_mean(data_in, 1, (size_t)npts);
        for (i=0; i<npts; i++) (x+d*npts)[i] = data_in[i] - mtmp + mean;
        
//        memcpy(x+d*npts, data_in, npts*sizeof(double));
    }
    
    // note we do not free the plan and the memory, for timing reasons
    return;
}


/****************************************************************************************/
/* creates a surrogate of a (possibly multi-dimensional) dataset of npts elements       */
/* in-place                                                                             */
/* "windowed Fourier transform" from Theiler et al Physica D 58 (1992) 2.4.2            */
/*                                                                                      */
/* 2023-02-27 : new function                                                            */
/* 2023-02-28 : tested OK.                                                              */
/* 2023-03-01 : added trick from Theiler himself                                        */
/****************************************************************************************/
void surrogate_wFt(double *x, int npts, int mx)
{   double u,v, mean, mtmp;
    int d, i;
    double eps=0.0001;
    
    init_surrogate_FFTW(npts);
    init_surrogates_rng();
    
    // create random phase shifts:
    for (i=0; i<(npts/2+1); i++) phase_shift[i]=gsl_rng_uniform(surrogates_rng);
    
    for (d=0; d<mx; d++)
    {   memcpy(data_in, x+d*npts, npts*sizeof(double));
        mean = gsl_stats_mean(data_in, 1, (size_t)npts);
        fftw_execute(plan_surrogate_direct);
        
        // windowing: 
//        for (i=0; i<npts; i++) data_in[i] *= (sin(2*M_PI*i/npts - M_PI/2)+1+eps)/2; // Vincent Croquette
        for (i=0; i<npts; i++) data_in[i] *= (sin(M_PI*i/npts) + eps)/2;    // James Theiler
        fftw_execute(plan_surrogate_direct);
        
        // randomize the phase by applying random phase shifts:
        for (i=0; i<(npts/2+1); i++)
	    {	u = data_Ft[i][0];
	        v = data_Ft[i][1];
	        data_Ft[i][0] = (u*cos(phase_shift[i]) - v*sin(phase_shift[i]))/npts;
		    data_Ft[i][1] = (u*sin(phase_shift[i]) + v*cos(phase_shift[i]))/npts;
	    }
	    // note that /npts is for the FFTw normalization, and it is speed-efficient to do it at this stage
	    
	    // trick from Theiler:
	    data_Ft[1][0]=0.;
	    data_Ft[1][1]=0.;
	    
        fftw_execute(plan_surrogate_inverse);
        // un-windowing:
//        for (i=0; i<npts; i++) data_in[i] /= (sin(2*M_PI*i/npts - M_PI/2)+1+eps)/2; // Vincent Croquette
        for (i=0; i<npts; i++) data_in[i] /= (sin(M_PI*i/npts) + eps)/2;      // James Theiler
        
        // 2023-03-01: added the following:
        mtmp = gsl_stats_mean(data_in, 1, (size_t)npts);
        for (i=0; i<npts; i++) (x+d*npts)[i] = data_in[i] - mtmp + mean;
//      memcpy(x+d*npts, data_in, npts*sizeof(double));
    }
    
    // note we do not free the plan and the memory, for timing reasons
    return;
}


/****************************************************************************************/
/* creates a Gaussian version of a dataset of npts elements                             */
/* in-place                                                                             */
/*                                                                                      */
/* dataset x is replaced by a Gaussian version of itself (with same PSD, etc)           */
/*                                                                                      */
/* see "amplitude adjusted Fourier transform" from Theiler et al Physica D 58 (1992)    */
/*                                                                                      */
/* 2023-02-28 : new function, tested OK.                                                */
/****************************************************************************************/
void Gaussianize(double *x, int npts, int mx)
{   int d=0, i;

    init_surrogate_FFTW(npts);
    init_surrogates_rng();
    surrogates_indices=(int*)calloc((size_t)npts, sizeof(int)); 
    surrogates_ind_inv=(int*)calloc((size_t)npts, sizeof(int)); 

    for (d=0; d<mx; d++)
    {   memcpy(data_in, x+d*npts, npts*sizeof(double)); // first dimension
        for (i=0; i<npts; i++)	surrogates_indices[i] = i;
	    QuickSort_double(data_in, surrogates_indices, 0, npts-1); /* we sort the data */
        for (i=0; i<npts; i++)	surrogates_ind_inv[surrogates_indices[i]] = i;    
        
        for (i=0; i<npts; i++)	data_in[i]=gsl_ran_gaussian(surrogates_rng, 1.); // std=1
        for (i=0; i<npts; i++)	surrogates_indices[i] = i;
	    QuickSort_double(data_in, surrogates_indices, 0, npts-1); /* we sort the data */
        for (i=0; i<npts; i++)	(x+d*npts)[i] = data_in[surrogates_ind_inv[i]];
    }
    
    free(surrogates_indices);    free(surrogates_ind_inv);
    return;
}

     
/****************************************************************************************/
/* creates a surrogate of a (possibly multi-dimensional) dataset of npts elements       */
/* in-place                                                                             */
/* "amplitude adjusted Fourier transform" from Theiler et al Physica D 58 (1992) 2.4.3  */
/*                                                                                      */
/* 2023-02-28 : new function, not optimized                                             */
/****************************************************************************************/
void surrogate_aaFt(double *x, int npts, int mx)
{   double u,v;
    int d=0, i;
    double *y;
    
    init_surrogate_FFTW(npts);
    init_surrogates_rng();
    surrogates_indices=(int*)calloc((size_t)npts, sizeof(int)); 
    surrogates_ind_inv=(int*)calloc((size_t)npts, sizeof(int)); 
    y              =(double*)calloc((size_t)npts, sizeof(double)); 

    // create random phase shifts:
    for (i=0; i<(npts/2+1); i++) phase_shift[i]=gsl_rng_uniform(surrogates_rng);
   
    for (d=0; d<mx; d++)
    {   memcpy(data_in, x+d*npts, npts*sizeof(double)); 
        for (i=0; i<npts; i++)	surrogates_indices[i] = i;
	    QuickSort_double(data_in, surrogates_indices, 0, npts-1); 
        for (i=0; i<npts; i++)	surrogates_ind_inv[surrogates_indices[i]] = i;    
        
        for (i=0; i<npts; i++)	y[i]=gsl_ran_gaussian(surrogates_rng, 1.); // std=1
        for (i=0; i<npts; i++)	surrogates_indices[i] = i;
	    QuickSort_double(y, surrogates_indices, 0, npts-1); 
        for (i=0; i<npts; i++)	data_in[i] = y[surrogates_ind_inv[i]];    // we now have y(t), a Gaussian version of the data
    
        // create surrogate y'(t):
        fftw_execute(plan_surrogate_direct);
        for (i=0; i<(npts/2+1); i++)        // randomize the phase by applying random phase shifts:
	    {	u = data_Ft[i][0];
	        v = data_Ft[i][1];
	        data_Ft[i][0] = (u*cos(phase_shift[i]) - v*sin(phase_shift[i]))/npts;
		    data_Ft[i][1] = (u*sin(phase_shift[i]) + v*cos(phase_shift[i]))/npts;
	    } 
        fftw_execute(plan_surrogate_inverse);       // we now have y'(t), a Gaussian Ft surrogate of the data

        //then reorder x to obtain the surrogate x':
        for (i=0; i<npts; i++)	surrogates_indices[i] = i;
	    QuickSort_double(data_in, surrogates_indices, 0, npts-1); /* we sort the data */
        for (i=0; i<npts; i++)	surrogates_ind_inv[surrogates_indices[i]] = i;    
        
        memcpy(y, x+d*npts, npts*sizeof(double));   // original x
        QuickSort_double(y, surrogates_indices, 0, npts-1); // order x, don't care about indices
        for (i=0; i<npts; i++)	(x+d*npts)[i] = y[surrogates_ind_inv[i]]; // re-ordered according to y'
    }
    
    free(y);
    free(surrogates_indices);    free(surrogates_ind_inv);
    return;
}





/****************************************************************************************/
/* creates a surrogate of a (possibly multi-dimensional) dataset of npts elements       */
/* in-place                                                                             */
/* "improved surrogate data" from Schreiber and Schmitz PRL 77 pp635-638 (1996)         */
/*                                                                                      */
/* 2023-02-28 : new function, untested                                                  */
/****************************************************************************************/
void surrogate_improved(double *x, int npts, int mx, int N_steps)
{   int d=0, i, i_step;
    double *y, *x_PSD, mod;
    
    init_surrogate_FFTW(npts);
    init_surrogates_rng();
    surrogates_indices=(int*)calloc((size_t)npts, sizeof(int)); 
    surrogates_ind_inv=(int*)calloc((size_t)npts, sizeof(int)); 
    y              =(double*)calloc((size_t)npts, sizeof(double)); 
    x_PSD          =(double*)calloc((size_t)npts/2+1, sizeof(double));
   
    for (d=0; d<mx; d++)
    {   memcpy(data_in, x+d*npts, npts*sizeof(double)); 
        fftw_execute(plan_surrogate_direct);
        // backup PSD of x:
        for (i=0; i<(npts/2+1); i++)    
        {   x_PSD[i] = sqrt(data_Ft[i][0]*data_Ft[i][0] + data_Ft[i][1]*data_Ft[i][1]);
        }
        // backup rank PDF of x:
        memcpy(y, x+d*npts, npts*sizeof(double));
        for (i=0; i<npts; i++)	surrogates_indices[i] = i;
	    QuickSort_double(y, surrogates_indices, 0, npts-1); /* we sort the data */
        for (i=0; i<npts; i++)	surrogates_ind_inv[surrogates_indices[i]] = i;    
        
        // step 0:
        gsl_ran_shuffle(surrogates_rng, data_in, npts, sizeof(double));     
        
        for (i_step=0; i_step<N_steps; i_step++)
        {   
            // first, we adjust the PSD:
            fftw_execute(plan_surrogate_direct);
            for (i=0; i<(npts/2+1); i++) 
            {   mod = sqrt(data_Ft[i][0]*data_Ft[i][0] + data_Ft[i][1]*data_Ft[i][1]);
                data_Ft[i][0] *= x_PSD[i]/mod/npts;
                data_Ft[i][1] *= x_PSD[i]/mod/npts;
            }
            fftw_execute(plan_surrogate_inverse);
            
            // second, we rank order:
            memcpy(y, data_in, npts*sizeof(double));  
            for (i=0; i<npts; i++)	surrogates_indices[i] = i;            // useless
	        QuickSort_double(y, surrogates_indices, 0, npts-1); 
            for (i=0; i<npts; i++)	data_in[i] = y[surrogates_ind_inv[i]]; // re-ordered according to original x    
          
        }
        
        memcpy(x+d*npts, data_in, npts*sizeof(double)); 
    }
    
    free(y);        free(x_PSD);
    free(surrogates_indices);    free(surrogates_ind_inv);
    return;
}
     
     
     
    
    
/****************************************************************************************/
/* init the random number generator foor shuffling                                      */
/****************************************************************************************/
void init_surrogates_rng(void)
{   const gsl_rng_type *T;

    if (surrogates_rng_state==0)
    {   gsl_rng_env_setup();
        T = gsl_rng_default;
        surrogates_rng = gsl_rng_alloc(T);
        surrogates_rng_state++;
    }
}


/****************************************************************************************/
/* shuffles a (possibly multi-dimensional) dataset of npts elements, in-place           */
/*                                                                                      */
/* 2023-01-30 : first version                                                           */
/* 2023-02-06 : this is the version used in cython, due to memory leaks with the other  */
/*              function below                                                          */
/****************************************************************************************/
void shuffle_data(double *x, int npts, int mx)
{   gsl_permutation *perm; // we need a permutation if mx>1
    double *tmp;
    int d, i;
    
    init_surrogates_rng();

    if (mx==1)  // scalar data: simple
    {   gsl_ran_shuffle(surrogates_rng, x, (size_t)npts, sizeof(double));
    }
    else  // we have to shuffle multiple dimensions altogether : we use a permutation 
    {   perm = gsl_permutation_alloc(npts);
        gsl_permutation_init(perm);
        gsl_ran_shuffle(surrogates_rng, perm->data, perm->size, sizeof(size_t));
        
        tmp  = (double*)calloc(npts, sizeof(double)); 
        // do the stuff:
        for (d=0; d<mx; d++)
        {   memcpy(tmp, x+d*npts, npts*sizeof(double));
            for (i=0; i<npts; i++)  (x+d*npts)[i] = tmp[perm->data[i]];
        }
        free(tmp);
            
        // free the permutation:
        gsl_permutation_free(perm);
    }
    
    return;
}


/****************************************************************************************/
/* creates a surrogate of a (possibly multi-dimensional) dataset of npts elements.      */
/* this surrogate is created by shuffling points in time to destroy time-dependences    */
/* (while keeping coordinates together)                                                 */
/*                                                                                      */
/* a new pointer is created on the fly (with the same dimensions as the input x)        */
/* and returned by the function                                                         */
/*                                                                                      */
/* this function should be slightly more efficient than creating a copy of the data,    */
/* and then shuffling it in-place, because we save some copy time and some malloc       */
/* (especially in mx>1 dimensions, and especially when in Python)                       */
/*                                                                                      */
/* 2023-01-31 : first version                                                           */
/****************************************************************************************/
double *create_surrogate(double *x, int npts, int mx)
{   gsl_permutation *perm; // we need a permutation if mx>1
    double *tmp;
    int d, i;
    
    init_surrogates_rng();
    tmp  = (double*)calloc(npts*mx, sizeof(double)); 
    
    if (mx==1)   
    {   memcpy(tmp, x, npts*sizeof(double));
        gsl_ran_shuffle(surrogates_rng, tmp, npts, sizeof(double));
    }
    else // we have to shuffle multiple dimensions altogether : we use a permutation 
    {   perm = gsl_permutation_alloc(npts);
        gsl_permutation_init(perm);
        gsl_ran_shuffle(surrogates_rng, perm->data, perm->size, sizeof(size_t));

        // do the stuff:
        for (d=0; d<mx; d++)
        {   for (i=0; i<npts; i++)  (tmp+d*npts)[i] = x[perm->data[i]+d*npts];
        }
            
        // free the permutation:
        gsl_permutation_free(perm);
    }
    return(tmp);
}

