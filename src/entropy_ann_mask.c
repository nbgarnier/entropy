/*
 *  entropy_ann_mask.c
 *  
 *  Created by Nicolas Garnier on 2013/06/20.
 *  Copyright 2013 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2013-06-20 : new source file containing all functions related to masking and epochs
 *  2013-06-21 : new function "compute_directed_information_ann_mask"
 *  2016-05-17 : bug correction and optimisation in calling "analyse_mask"
 *  2019-01-24 : changed T_min definition in all functions, and improved testing
 *  2019-01-25 : renamed compute_*_nd_ann functions as compute_*_ann
 *  2021-12-15 : multithread versions of all functions (but untested)
 *  2022-06-10 : new samplings for all functions (most untested)
 *  2023-02-23 : TE(X->Y) is now correctly ordered (was: TE(Y->X)); same with DI
 */

#include <math.h>                        // for fabs
#include <string.h>
#include <gsl/gsl_statistics_double.h>   // for std of intermediate (pre-processed) data

#include "library_commons.h"             // for definition of nb_errors, and stds
#include "library_matlab.h"              // compilation for Matlab
#include "entropy_ann.h"                 // for "front-end" functions 
#include "entropy_ann_mask.h"
#include "mask.h"                        // for basic masking functions
#include "samplings.h"                   // for sub-sampling functions
#include "entropy_ann_single_entropy.h"  // for "engine" function for Shannon entropy
#include "entropy_ann_single_RE.h"       // for "engine" function for relative entropy
#include "entropy_ann_single_MI.h"       // for "engine" function for MI 
#include "entropy_ann_single_PMI.h"      // for "engine" function for PMI
#include "entropy_ann_threads.h"         // for multithread management functions
#include "entropy_ann_threads_entropy.h" // for "engine" function for Shannon entropy with pthreads
#include "entropy_ann_threads_Renyi.h"   // for "engine" function for Renyi entropy with pthreads
#include "entropy_ann_threads_RE.h"      // for "engine" function for relative entropy 
#include "entropy_ann_threads_MI.h"      // for "engine" function for MI with pthreads
#include "entropy_ann_threads_PMI.h"     // for "engine" function for PMI with pthreads
#include "entropy_ann_Renyi.h"           // for "engine" function for Renyi entropy
#include "timings.h"

#define noDEBUG		// for debug information, replace "noDEBUG" by "DEBUG"
#define LOOK 167	// for debug also (of which point(s) will we save the data ?)



/****************************************************************************************/
/* computes Shannon entropy, using nearest neighbor statistics (Grassberger 2004)       */
/*																			            */
/* this version is for m-dimentional systems, with eventually some stride/embedding	    */
/*              and uses a mask to define epochs of interest                            */
/*																			            */
/* x        contains all the data, which is of size nx in time						    */
/* mask     contains the mask (values 0 or 1 along time) to define epochs               */
/* npts     is the number of points in time											    */
/* m	    indicates the (initial) dimensionality of x								    */
/* p	    indicates how many points to take in time (in the past) (embedding)         */
/* stride   is the time lag between 2 consecutive points to be considered in time		*/
/* k        nb of neighbors to be considered										    */
/* method   0 for regular entropy                                                       */
/*          1 for entropy of the increments                                             */
/*          2 for entropy of the averaged increments                                    */
/*																			            */
/* data is ordered like this :													        */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)				*/
/*																			            */
/* this function is a  wrapper to the functions :									    */
/*	- compute_entropy_nd_ann													        */
/*																			            */
/* 2013-06-19 : new functions, forked from "compute_entropy_ann"                        */
/* 2019-01-24 : this function is very conservative, more windows can be used!           */
/*              idea : moving mask to follow the windowing                              */
/* 2019-01-24 : function now returns n_sets_total (nb of used points accross epochs)    */
/* 2020-02-26 : nb of errors is returned in the global variable "nb_errors"             */
/* 2021-12-15 : pthreads                                                                */
/* 2022-05-28 : new samplings (WIP)                                                     */
/* 2022-06-03 : this function is merged with _increments(), and "method" is introduced  */
/* 2022-06-06 : new samplings working OK                                                */
/****************************************************************************************/
int compute_entropy_ann_mask(double *x, char *mask, int npts, int m, int p, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S)
{	register int i, j, p_new;
	int     npts_good, N_real_max, n_sets_total=0;
	double *x_new, avg=0.0, var=0.0, S_tmp=0.0;
	double x_new_std=0.;    // 2022-06-06: for the std of the pre-processed signal
    int     *ind_epoch=NULL;
    size_t  *ind_shuffled=NULL;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations, .type=-1};
	gsl_permutation *perm_pts;
//	double temps;
	
    *S = my_NAN;
        
	// adapt embedding dimension according to method (2025-02-17, moved before tests (on p))
	if (method==0)  p_new = p;  // time-embedding (not increments)
	else {          p_new = 1;  // increments (regular or averaged)
	                p    += 1;  // convention: increments of order 1 (p) require 2 (p+1) points in order to be computed
	     }
    
    if ((m<1) || (p<1)) return(printf("[compute_entropy_ann_mask] : m and p must be at least 1 !\n"));
    if (k<1)            return(printf("[compute_entropy_ann_mask] : k must be at least 1 !\n"));
    if (stride<1)       return(printf("[compute_entropy_ann_mask] : stride must be at least 1 !\n"));
    if ((method<0) || (method>2))
                        return(printf("[compute_entropy_ann_mask] : method must be 0, 1 or 2 !\n"));
 
    N_real_max = set_sampling_parameters_mask(mask, npts, p, stride, 0, &sp, "compute_entropy_ann_mask");
    if (N_real_max<1)   return(printf("[compute_entropy_ann_mask] : aborting\n"));
    if (sp.N_eff < 2*k) return(printf("[compute_entropy_ann_mask] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
//printf("N_real_max = %d / N_eff = %d\n", sp.N_real_max, sp.N_eff);

    x_new    = (double*)calloc(m*p_new*sp.N_eff, sizeof(double));
    
    nb_errors=0; last_npts_eff=0;   // global variables
    data_std=0;  data_std_std=0;    // 2022-03-11, for std of the increments
    for (j=0; j<sp.N_real; j++)     // loop over independant windows
    {   
//temps=0; tic();   
        npts_good = analyze_mask_for_sampling(mask+j, npts-j, p, stride, 0, sp.Theiler, 1, &ind_epoch); 
//toc(&temps);        
//if (j==0) printf("stride %d, Theiler %d, N_eff_max %d, time to analyze mask: %f", stride, sp.Theiler, sp.N_eff_max, temps);
        
        if (npts_good<sp.N_eff) 
        {   printf("[compute_entropy_ann_mask] : npts_good = %d < N_eff =%d!!!\n", npts_good, sp.N_eff);
            return(-1);
        }
//temps=0; tic();       
        perm_pts  = create_unity_perm(npts_good); shuffle_perm(perm_pts); // for random sampling
        ind_shuffled = (size_t*)calloc(npts_good, sizeof(size_t));
//toc(&temps);        
//if (j==0) printf("\ttime for perm creation and shuffling: %f (%d pts)\n", temps, npts_good);
        for (i=0; i<npts_good; i++) ind_shuffled[i] = ind_epoch[perm_pts->data[i]]; // OK
                    
        // note: in the calls below, the shift must not include (p-1)*stride, because it is already
        // accounted for in the indices in 'ind_shuffled'
        if (method==0)      // regular embedding (not increments)
            Theiler_embed_mask(x+j, npts, m, p, stride, ind_shuffled, x_new, sp.N_eff);
        else if (method==1) // regular increments
            increments_mask   (x+j, npts, m, p, stride, ind_shuffled, x_new, sp.N_eff);
        else if (method==2) // averaged increments
            incr_avg_mask     (x+j, npts, m, p, stride, ind_shuffled, x_new, sp.N_eff);
    
        // 2022-06-06: std of the increments:
        x_new_std     = gsl_stats_sd(x_new, 1, sp.N_eff);
        data_std     += x_new_std;
        data_std_std += x_new_std*x_new_std;
    
        if (USE_PTHREAD>0) S_tmp = compute_entropy_nd_ann_threads(x_new, sp.N_eff, m*p_new, k,
                                    get_cores_number(GET_CORES_SELECTED));
        else               S_tmp = compute_entropy_nd_ann        (x_new, sp.N_eff, m*p_new, k);

        avg  += S_tmp;
        var  += S_tmp*S_tmp;
        nb_errors += nb_errors_local; // each call to "compute_entropy_nd_ann" gives a new value of nb_errors_local
        last_npts_eff += last_npts_eff_local;
        
        free(ind_epoch);
        free(ind_shuffled);
        free_perm(perm_pts);
        n_sets_total += sp.N_eff;
    }
    avg /= sp.N_real;
    var /= sp.N_real;   var -= avg*avg;
    data_std     /= sp.N_real; 
    data_std_std /= sp.N_real;  data_std_std -= data_std*data_std;
    
    *S = avg;
    last_std = sqrt(var);
    nb_errors_total += nb_errors;
    last_samp=sp;
    
    free(x_new);
	return(n_sets_total);   // 2020-02-26: now, nb of errors is returned in global variable "nb_errors" 
} /* end of function "compute_entropy_ann_mask" *************************************/



/****************************************************************************************/
/* computes Renyi entropy, using nearest neighbor statistics (Leonenko 2008)            */
/*                                                                                      */
/* this version is for m-dimentional systems, with eventually some stride/embedding     */
/*                                                                                      */
/* x      contains all the data, which is of size nx in time                            */
/* mask   contains the mask (values 0 or 1 along time) to define epochs                 */
/* npts   is the number of points in time                                               */
/* m      indicates the (initial) dimensionality of x                                   */
/* p      indicates how many points to take in time (in the past) (embedding)           */
/* stride is the time lag between 2 consecutive points to be considered in time         */
/* q      is the order of the Renyi entropy                                             */
/* k      nb of neighbors to be considered                                              */
/*                                                                                      */
/* data is ordered like this :                                                          */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)               */
/*                                                                                      */
/* this function is a  wrapper to the functions :                                       */
/*    - compute_Renyi_nd_nns                                                            */
/*                                                                                      */
/* 2019-02-05 : first version, forked from compute_Renyi_ann                            */
/* 2020-02-26 : function now returns n_sets_total (nb of used points accross epochs)    */
/* 2020-02-26 : nb of errors is returned in the global variable "nb_errors"             */
/* 2021-12-15 : pthreads                                                                */
/****************************************************************************************/
int compute_Renyi_ann_mask(double *x, char *mask, int npts, int m, int p, int stride, double q, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S)
{   register int i,j, p_new;
    int    npts_good, N_real_max, n_sets_total=0;
    double *x_new, avg=0.0, var=0.0, S_tmp=0.0;
    double x_new_std=0.;    // 2022-06-08: for the std of the pre-processed signal
    int     *ind_epoch=NULL;
    size_t  *ind_shuffled=NULL;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations, .type=-1};
	gsl_permutation *perm_pts;

    *S = my_NAN;

    // adapt embedding dimension according to method:
	if (method==0)  p_new = p;  // time-embedding (not increments)
	else {          p_new = 1;  // increments (regular or averaged)
	                p    += 1;  // convention: increments of order 1 (p) require 2 (p+1) points in order to be computed
	     }

    if ((m<1) || (p<1))     return(printf("[compute_Renyi_ann_mask] : m and p must be at least 1 !\n"));
    if (k<1)                return(printf("[compute_Renyi_ann_mask] : k must be at least 1 !\n"));
    if (stride<1)           return(printf("[compute_Renyi_ann_mask] : stride must be at least 1 !\n"));
    if (q==1)               return(printf("[compute_Renyi_ann_mask] : Renyi entropy of order q=1?...\n"));
    if ((method<0) || (method>2))
                        return(printf("[compute_entropy_ann_mask] : method must be 0, 1 or 2 !\n"));
    
    N_real_max = set_sampling_parameters_mask(mask, npts, p, stride, 0, &sp, "compute_entropy_ann_mask");
    if (N_real_max<1)   return(printf("[compute_entropy_ann_mask] : aborting\n"));
    if (sp.N_eff < 2*k) return(printf("[compute_entropy_ann_mask] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));

    x_new    = (double*)calloc(m*p_new*sp.N_eff, sizeof(double));

    nb_errors=0; last_npts_eff=0; // global variables
    data_std=0;  data_std_std=0;    // 2022-03-11, for std of the increments
    for (j=0; j<sp.N_real; j++)   // loop over independant windows
    {   npts_good = analyze_mask_for_sampling(mask+j, npts-j, p, stride, 0, sp.Theiler, 1, &ind_epoch); 
        if (npts_good<sp.N_eff) 
        {   printf("\tnpts_good = %d < N_eff =%d!!!\n", npts_good, sp.N_eff);
            return(-1);
        }
        
        perm_pts  = create_unity_perm(npts_good); shuffle_perm(perm_pts); // for random sampling
        ind_shuffled = (size_t*)calloc(npts_good, sizeof(size_t));
        for (i=0; i<npts_good; i++) ind_shuffled[i] = ind_epoch[perm_pts->data[i]]; // OK
        
        // note: in the calls below, the shift must not include (p-1)*stride, because it is already
        // accounted for in the indices in 'ind_shuffled'
        if (method==0)      // regular embedding (not increments)
            Theiler_embed_mask(x+j, npts, m, p, stride, ind_shuffled, x_new, sp.N_eff);
        else if (method==1) // regular increments
            increments_mask   (x+j, npts, m, p, stride, ind_shuffled, x_new, sp.N_eff);
        else if (method==2) // averaged increments
            incr_avg_mask     (x+j, npts, m, p, stride, ind_shuffled, x_new, sp.N_eff);
    
        // 2022-06-06: std of the increments:
        x_new_std     = gsl_stats_sd(x_new, 1, sp.N_eff);
        data_std     += x_new_std;
        data_std_std += x_new_std*x_new_std;
    
        if (USE_PTHREAD>0) S_tmp = compute_Renyi_nd_ann_threads(x_new, sp.N_eff, m*p_new, q, k,
                                    get_cores_number(GET_CORES_SELECTED));
        else               S_tmp = compute_Renyi_nd_ann        (x_new, sp.N_eff, m*p_new, q, k);
        
        avg  += S_tmp;
        var  += S_tmp*S_tmp;
        nb_errors += nb_errors_local; // each call to "compute_Renyi_nd_ann" gives a new value of nb_errors_local
        last_npts_eff += last_npts_eff_local;
        
        free(ind_epoch);
        free(ind_shuffled);
        free_perm(perm_pts);
        n_sets_total += sp.N_eff;
    }
    avg /= sp.N_real;
    var /= sp.N_real;   var -= avg*avg;
    data_std     /= sp.N_real; 
    data_std_std /= sp.N_real;  data_std_std -= data_std*data_std;
    
    *S = avg;
    last_std = sqrt(var);
    nb_errors_total += nb_errors;
    last_samp=sp;
    
    free(x_new);
    return(n_sets_total);
} /* end of function "compute_Renyi_ann_mask" *******************************************/




/****************************************************************************************/
/* computes Shannon entropy rate, using nearest neighbor statistics                     */
/*                                                                                      */
/* this version is for m-dimentional systems, with eventually some stride/embedding     */
/*                                                                                      */
/* x      contains all the data, which is of size nx in time                            */
/* nx     is the number of points in time                                               */
/* m      indicates the (initial) dimensionality of x                                   */
/* p      indicates how many points to take in time (in the past) (embedding)           */
/* stride is the time lag between 2 consecutive points to be considered in time         */
/* k      nb of neighbors to be considered                                              */
/* method is the method to use : 0 for fraction, 1 for difference and 2 for H-MI        */
/*                                                                                      */
/* data is ordered like this :                                                          */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)               */
/*                                                                                      */
/* this function uses the functions :                                                   */
/*    - compute_entropy_ann                                                             */
/*    - compute_mutual_information_ann                                                  */
/*                                                                                      */
/* 2019-01-26: 3 methods                                                                */
/* 2020-02-23: bug corrected in method=2                                                */
/* 2020-02-24: forked from "compute_entropy_rate_ann"                                   */
/****************************************************************************************/
int compute_entropy_rate_ann_mask(double *x, char *mask, int npts, int m, int p, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S)
{   register int i, d;
    double   H_past=0.0, H=0.0, I1=0.0, I2=0.0;
    char    *new_mask=NULL;
    double  *x_past, *x_now;
    int  my_nb_errors=0; // 2020-02-26: we need local counting, because we invoke here several "direct" functions
    int  my_npts_eff =0; // 2020-02-28: we need local counting, because we invoke here several "direct" functions
    
    *S = my_NAN;    // default returned value
    
    if ((method!=ENTROPY_RATE_FRACTION) && (method!=ENTROPY_RATE_DIFFERENCE) && (method!=ENTROPY_RATE_MI))
                        return(printf("[compute_entropy_rate_ann] : invalid method (should be 0,1 or 2)!\n"));
    if ((m<1) || (p<1)) return(printf("[compute_entropy_rate_ann] : m and p must be at least 1 !\n"));
    if ((stride<1))     return(printf("[compute_entropy_rate_ann] : stride must be at least 1 !\n"));
    if ((k<1))          return(printf("[compute_entropy_rate_ann] : k must be at least 1 !\n"));
    
    if (method==ENTROPY_RATE_FRACTION)
    {   compute_entropy_ann_mask(x, mask, npts, m, p, stride, tau_Theiler, N_eff, N_realizations, k, 0, &H);
        my_nb_errors += nb_errors; // 2020-02-26: new way of getting the nb of errors returned by a "direct" function 
        my_npts_eff  += last_npts_eff;
        *S = H/p;
    }
    if (method==ENTROPY_RATE_DIFFERENCE)
    {   x_past = calloc( (npts-stride)*m , sizeof(double));
        x_now  = x;
        
        for (d=0; d<m; d++)
        for (i=0; i<npts-stride; i++)
        {   x_past[i + d*(npts-stride)] = x[i+d*npts];
        }
        compute_entropy_ann_mask(x_past, mask, npts-stride, m, p,   stride, tau_Theiler, N_eff, N_realizations, k, 0, &H_past);
//      my_nb_errors += nb_errors; my_npts_eff += last_npts_eff;
        compute_entropy_ann_mask(x_now,  mask, npts,        m, p+1, stride, tau_Theiler, N_eff, N_realizations, k, 0, &H);
//      my_nb_errors += nb_errors; my_npts_eff += last_npts_eff;
        *S = H-H_past;
        
        free(x_past);
    }
    if (method==ENTROPY_RATE_MI) // 2020-02-23: untested!!! masking has to apply on 2 pointers (x and x+stride)
    {   x_past   = calloc( (npts-stride)*m , sizeof(double));
        x_now    = calloc( (npts-stride)*m , sizeof(double));
        new_mask = calloc( (npts-stride)   , sizeof(char));
        
        for (d=0; d<m; d++)
        for (i=0; i<npts-stride; i++)
        {   x_past[i + d*(npts-stride)] = x[i        + d*npts];
            x_now [i + d*(npts-stride)] = x[i+stride + d*npts];
        }
        combine_masks(mask+stride, mask, npts-stride, new_mask); 
        // 2020-02-23: check how embedding is performed in MI function below!!!
        
//        compute_mutual_information_ann_mask(x+stride, x, new_mask, npts-stride, m, m, 1, p, stride, tau_Theiler, N_eff, N_realizations, k, &I1, &I2);
//      my_nb_errors += nb_errors; my_npts_eff += last_npts_eff;

        compute_entropy_ann_mask           (x_now,         new_mask, npts-stride, m,    1,    stride, tau_Theiler, N_eff, N_realizations, k, 0,   &H);
        compute_mutual_information_ann_mask(x_now, x_past, new_mask, npts-stride, m, m, 1, p, stride, tau_Theiler, N_eff, N_realizations, k, &I1, &I2); 
        my_nb_errors += nb_errors; my_npts_eff += last_npts_eff;
        
        *S = (MI_algo&MI_ALGO_1) ? H-I1 : H-I2;
        
        free(x_past); free(x_now); free(new_mask);
    }
    
    nb_errors     = my_nb_errors;
    last_npts_eff = my_npts_eff;
    return(nb_errors);
} /* end of function "compute_entropy_rate_ann_mask" ************************************/





/****************************************************************************************/
/* computes mutual information, using nearest neighbor statistics                       */
/* this is an application of Grassberger PRE 69 066138 (2004)		    				*/
/*																			            */
/* this version computes information redundency of 2 variables x and y                  */
/* of initial dimensions mx and my, after embedding of px points in x and py in y       */
/* final system is (mx*px + my*py)-dimensional                                          */
/*																			            */
/* This function allows the use of a mask, to keep only given epochs of the signal      */
/*                                                                                      */
/*																			            */
/* x,y  contains all the datasets, which are of size nx in time						    */
/* mask contains the mask (1 for times to keep, 0 otherwise)                            */
/* npts is the number of points in time											        */
/* mx   is the nb of dimension of x before embedding                                    */
/* my   is the nb of dimension of y before embedding                                    */
/* px   indicates how many points to take in the past of x	(embedding)				    */
/* py   indicates how many points to take in the past of y 	(embedding)                 */
/* stride is the time lag between 2 consecutive points in time when embedding			*/
/*																			            */
/* this function is a wrapper to the function :									        */
/*	- compute_mutual_information_2xnd_ann										        */
/*																			            */
/* 2013-06-18: first version, forked  from "compute_mutual_information_nd_ann_mask"     */
/*              after half a day of intense fight, this has been tested OK              */
/* 2020-02-24: checkedd that the embedding is causal                                    */
/* 2021-12-15: multithreading enabled                                                   */
/* 2023-12-01: bug correction (wrong pointer as parameter of "Theiler_embed_mask")      */
/****************************************************************************************/
int compute_mutual_information_ann_mask(double *x, double *y, char *mask, int npts, int mx, int my, int px, int py, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)
{	double *x_new;
	double mi1=0.0, mi2=0.0, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;
	register int i, j;
	int     pp = (px>py) ? px : py;	  // largest past (largest embedding dimension)
	int     n  = mx*px + my*py;
    int     N_real_max=0, npts_good;	
    int     *ind_epoch=NULL;
    size_t  *ind_shuffled=NULL;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations, .type=-1};
	gsl_permutation *perm_pts;
	
    // default value (in case of return on error):	
    *I1=my_NAN; *I2=my_NAN;

	if ((mx<1) || (my<1)) return(printf("[compute_mutual_information_ann_mask] : mx and my must be at least 1 !\n"));
	if ((px<1) || (py<1)) return(printf("[compute_mutual_information_ann_mask] : px and py must be at least 1 !\n"));
	if (stride<1)         return(printf("[compute_mutual_information_ann_mask] : stride must be at least 1 !\n"));
    if (k<1)              return(printf("[compute_mutual_information_ann-mask] : k has to be at least 1.\n"));

    N_real_max = set_sampling_parameters_mask(mask, npts, pp, stride, 0, &sp, "compute_mutual_information_ann_mask");
    if (N_real_max<1)   return(printf("[compute_mutual_information_ann_mask] : aborting\n"));
    if (sp.N_eff < 2*k) return(printf("[compute_mutual_information_ann_mask] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
    
    x_new    = (double*)calloc(n*sp.N_eff, sizeof(double));
    
    nb_errors=0; last_npts_eff=0;
    for (j=0; j<sp.N_real; j++) // loop over independant windows (independant !!! cf Theiler !!!)
	{   
	    npts_good = analyze_mask_for_sampling(mask+j, npts-j, pp, stride, 0, sp.Theiler, 1, &ind_epoch); 
        if (npts_good<sp.N_eff) 
        {   printf("[compute_mutual_information_ann_mask] : npts_good = %d < N_eff =%d!!!\n", npts_good, sp.N_eff);
            return(-1);
        }      
        perm_pts  = create_unity_perm(npts_good); shuffle_perm(perm_pts); // for random sampling
        ind_shuffled = (size_t*)calloc(npts_good, sizeof(size_t));
        for (i=0; i<npts_good; i++) ind_shuffled[i] = ind_epoch[perm_pts->data[i]]; // OK
        
        // note: in the calls below, the shift must not include (p-1)*stride, because it is already
        // accounted for in the indices in 'ind_shuffled'
        Theiler_embed_mask(x+j, npts, mx, px, stride, ind_shuffled, x_new,                sp.N_eff);
        Theiler_embed_mask(y+j, npts, my, py, stride, ind_shuffled, x_new+mx*px*sp.N_eff, sp.N_eff);
        
        if (USE_PTHREAD>0)  // if we want multithreading
            nb_errors += compute_mutual_information_2xnd_ann_threads(x_new, sp.N_eff, 
                                mx*px, my*py, k, &mi1, &mi2, get_cores_number(GET_CORES_SELECTED));
        else                // single threaded algorithms:
            nb_errors += compute_mutual_information_direct_ann(x_new, sp.N_eff, mx*px, my*py, k, &mi1, &mi2);
  
        avg1 += mi1;    var1 += mi1*mi1;
        avg2 += mi2;	var2 += mi2*mi2;
		last_npts_eff += last_npts_eff_local;
		nb_errors += nb_errors_local; // each call gives a new value of nb_errors_local
        
        free(ind_epoch);
        free(ind_shuffled);
        free_perm(perm_pts);
    }
    
	avg1 /= sp.N_real;  var1 /= sp.N_real;  var1 -= avg1*avg1;
	avg2 /= sp.N_real;  var2 /= sp.N_real;  var2 -= avg2*avg2;
	*I1  = avg1;        last_std  = sqrt(var1);
	*I2  = avg2;		last_std2 = sqrt(var2);

    nb_errors_total += nb_errors;
	last_samp=sp;

	free(x_new);
	return(nb_errors); 
} /* end of function "compute_mutual_information_nd_ann_mask" **********************************/





/*************************************************************************************
 * to compute transfer entropy	"directly" from combinations of KL estimators
 * this version analyses the case of a point in the future at arbitrary distance "lag"
 *
 * this reproduces the combination used in article from Vicente, Wibral, Lindner, Pipa 
 *        J Comput Neurosci DOI 10.1007/s10827-010-0262-3
 *        "Transfer entropy — a model-free measure of effective connectivity for the neurosciences"
 *
 * TE(X->Y) : influence of X (1st argument) on Y (2nd argument)
 * TE(X->Y) = PMI(Y+lag, X | Y)
 *
 * x      contains the first variable (n-d)
 * y      contains the second variable (n-d)
 * mask   contains the mask defining epochs to use
 * nx     is the number of points in time	 (same for x and y)
 * mx     is the number of dimensions of x
 * my     is the number of dimensions of y
 * px     is the number of points to consider in the past of x (embedding)
 * py     is the number of points to consider in the past of y (embedding)
 * stride is the distance between 2 points in time, in the past
 * lag    is the distance between present x_n, and future point x_(n+lag)
 * k      is the number of neighbors to consider
 *
 * This is just a wrapper to the function "compute_partial_MI_direct_ann"
 *
 * 2013-06-19 : fork from function "compute_transfer_entropy_nd_ann"
 * 2013-06-20 : working version, with improved tests of sanity (on the nb of available points)
 * 2021-12-15 : pthreads
 * 2023-02-23 : now TE(X->Y) (1st -> 2nd argument) instead of the opposite
 * 2023-12-01 : bug correction (same as in MI)
 *************************************************************************************/
int compute_transfer_entropy_ann_mask(double *x, double *y, char *mask, int npts, int mx, int my, int px, int py, int stride, int lag, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *T1, double *T2)
{	register int i, j;
    register int nn	= my*(py+1)+mx*px;   // dimension of the new variable
    register int pp = (px>py) ? px : py; // largest past (largest embedding dimension)
	int     N_real_max=0, npts_good;	
	double  *x_new;
	double  te1=0.0, te2=0.0, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;
    int     *ind_epoch;
    size_t  *ind_shuffled=NULL;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations, .type=-1};
	gsl_permutation *perm_pts;

    // default value (in case of return on error):
    *T1=my_NAN; *T2=my_NAN;
    
	if (lag<1)            return(printf("[compute_transfer_entropy_ann_mask] : lag has to be equal or larger than 1.\n"));
	if (stride<1)         return(printf("[compute_transfer_entropy_ann_mask] : stride must be at least 1 !\n"));
    if ((mx<1) || (my<1)) return(printf("[compute_transfer_entropy_ann_mask] : mx and my must be larger than 1 !\n"));
    if ((px<1) || (py<1)) return(printf("[compute_transfer_entropy_ann_mask] : px and py must be larger than 1 !\n"));
    if (k<1)              return(printf("[compute_transfer_entropy_ann_mask] : k has to be equal or larger than 1.\n"));
    
    N_real_max = set_sampling_parameters_mask(mask, npts, pp, stride, lag, &sp, "compute_transfer_entropy_ann_mask");
    if (N_real_max<1)     return(printf("[compute_transfer_entropy_ann_mask] : aborting\n"));
    if (sp.N_eff < 2*k)   return(printf("[compute_transfer_entropy_ann_mask] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
    
    x_new  = (double*)calloc(nn*sp.N_eff, sizeof(double));

    nb_errors=0; last_npts_eff=0;
    for (j=0; j<sp.N_real; j++)         // loop over independant windows (independant !!! cf Theiler !!!)
    {   
        npts_good = analyze_mask_for_sampling(mask+j, npts-j, pp, stride, lag, sp.Theiler, 1, &ind_epoch); 
        if (npts_good<sp.N_eff) 
        {   printf("\tnpts_good = %d < N_eff =%d!!!\n", npts_good, sp.N_eff);
            return(-1);
        }      
        perm_pts  = create_unity_perm(npts_good); shuffle_perm(perm_pts); // for random sampling
        ind_shuffled = (size_t*)calloc(npts_good, sizeof(size_t));
        for (i=0; i<npts_good; i++) ind_shuffled[i] = ind_epoch[perm_pts->data[i]]; // OK
        
        // note: in the calls below, the shift must not include (p-1)*stride, because it is already
        // accounted for in the indices in 'ind_shuffled'
        Theiler_embed_mask(y+j+lag, npts, my,  1, stride, ind_shuffled, x_new,                    sp.N_eff);
        Theiler_embed_mask(y+j,     npts, my, py, stride, ind_shuffled, x_new+my*sp.N_eff,        sp.N_eff);
        Theiler_embed_mask(x+j,     npts, mx, px, stride, ind_shuffled, x_new+my*(py+1)*sp.N_eff, sp.N_eff);
       
/*        for (i=0; i<nx_new; i++)        // loop on time over acceptable points for the current window
        {   for (d=0; d<mx; d++)        // loop over dimensions in x (future point)
                x_new[i +                      d*nx_new] = x[ind_epoch[i] + d*npts + shift + lag ];
            for (d=0; d<mx; d++)        // loop over existing dimensions in x
            for (l=0; l<px; l++)        // loop over embedding  in x
                x_new[i +        (mx + d + l*mx)*nx_new] = x[ind_epoch[i] + d*npts + shift - stride*l];
            for (d=0; d<my; d++)        // loop over dimensions in y
            for (l=0; l<py; l++)        // loop over embedding  in y
                x_new[i + (mx*(px+1) + d + l*my)*nx_new] = y[ind_epoch[i] + d*npts + shift - stride*l];
        }
*/
        if (USE_PTHREAD>0)  // if we want multithreading
            nb_errors += compute_partial_MI_direct_ann_threads(x_new, sp.N_eff, my, mx*px, my*py, k, &te1, &te2, get_cores_number(GET_CORES_SELECTED));
        else                // single threaded algorithms:
            nb_errors += compute_partial_MI_engine_ann        (x_new, sp.N_eff, my, mx*px, my*py, k, &te1, &te2);

        avg1 += te1;    var1 += te1*te1;
		avg2 += te2;	var2 += te2*te2;
		last_npts_eff += last_npts_eff_local;
        nb_errors += nb_errors_local;
        
        free(ind_epoch);
        free(ind_shuffled);
        free_perm(perm_pts);
    } // loop on j
    
    avg1 /= sp.N_real;  var1 /= sp.N_real;  var1 -= avg1*avg1;
	avg2 /= sp.N_real;	var2 /= sp.N_real;	var2 -= avg2*avg2;
	*T1 = avg1;         last_std  = sqrt(var1);
	*T2 = avg2;         last_std2 = sqrt(var2);

    nb_errors_total += nb_errors;	
	last_samp=sp;

	free(x_new);
	return(nb_errors);
} /* end of function "compute_transfer_entropy_ann_mask" */





/*************************************************************************************
 * to compute partial transfer entropy "directly" from combinations of KL estimators
 * this version analyses the case of a point in the future at arbitrary distance "lag"
 *
 * PTE(X->Y|Z) : influence of X (1st argument) on Y (2nd argument), conditionned on Z (3rd argument)
 * PTE(X->Y|Z) = PMI(Y+lag, X | Y, Z)
 *
 * x      contains the first variable (n-d)
 * y      contains the second variable (n-d)
 * z      contains the conditioning variable (n-d)
 * mask   contains the mask defining epochs to use
 * nx     is the number of points in time	 (same for x, y and z)
 * mx     is the number of dimensions of x
 * my     is the number of dimensions of y
 * px     is the number of points to consider in the past of x (embedding)
 * py     is the number of points to consider in the past of y (embedding)
 * stride is the distance between 2 points in time, in the past
 * lag    is the distance between present x_n, and future point x_(n+lag)
 * k      is the number of neighbors to consider
 *
 * This is just a wrapper to the function "compute_partial_MI_direct_ann"
 *
 * 2013-07-19 : fork from function "compute_transfer_entropy_nd_ann_mask"
 *              untested
 * 2023-02-23 : now PTE(X->Y) (1st -> 2nd argument) instead of the opposite
 * 2023-12-01 : bug correction (same as in MI)
 *************************************************************************************/
int compute_partial_TE_ann_mask(double *x, double *y, double *z, char *mask, int npts, int *dim, int stride, int lag, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *T1, double *T2)
{	register int i, j, n, pp,
            mx=dim[0], my=dim[1], mz=dim[2], px=dim[3+0], py=dim[3+1], pz=dim[3+2];
	double *x_new;
	double te1=0.0, te2=0.0, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;
    int     N_real_max=0, npts_good;	
    int     *ind_epoch=NULL;
    size_t  *ind_shuffled=NULL;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations, .type=-1};
	gsl_permutation *perm_pts;
	    
    // default value (in case of return on error):
    *T1=my_NAN; *T2=my_NAN;
    
	if (lag<1)      return(printf("[compute_partial_TE_ann_mask] : lag has to be equal or larger than 1.\n"));
	if (stride<1)   return(printf("[compute_partial_TE_ann_mask] : stride must be at least 1 !\n"));
    if ((mx<1) || (my<1) || (mz<1))
                    return(printf("[compute_partial_TE_ann_mask] : initial dimensions of data must be at least 1 !\n"));
    if ((px<1) || (py<1) || (pz<1))
                    return(printf("[compute_partial_TE_ann_mask] : embedding dimensions must be at least 1 !\n"));
    if (k<1)        return(printf("[compute_partial_TE_ann_mask] : k has to be equal or larger than 1.\n"));
    
	n	 = my*(py+1)+mx*px+mz*pz;   // dimension of the new variable
	pp   = (px>py) ? px : py;       // who has the largest past ?
    pp   = (pp>pz) ? pp : pz;

	N_real_max = set_sampling_parameters_mask(mask, npts, pp, stride, lag, &sp, "compute_partial_TE_ann_mask");
    if (N_real_max<1)   return(printf("[compute_partial_TE_ann_mask] : aborting\n"));
    if (sp.N_eff < 2*k) return(printf("[compute_partial_TE_ann_mask] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
    
    x_new    = (double*)calloc(n*sp.N_eff, sizeof(double));
    
    nb_errors=0; last_npts_eff=0;
    for (j=0; j<sp.N_real; j++)         // loop over independant windows (independant !!! cf Theiler !!!)
    {   
        npts_good = analyze_mask_for_sampling(mask+j, npts-j, pp, stride, lag, sp.Theiler, 1, &ind_epoch); 
        if (npts_good<sp.N_eff) 
        {   printf("\tnpts_good = %d < N_eff =%d!!!\n", npts_good, sp.N_eff);
            return(-1);
        }      
        perm_pts  = create_unity_perm(npts_good); shuffle_perm(perm_pts); // for random sampling
        ind_shuffled = (size_t*)calloc(npts_good, sizeof(size_t));
        for (i=0; i<npts_good; i++) ind_shuffled[i] = ind_epoch[perm_pts->data[i]]; // OK

        // note: in the calls below, the shift must not include (p-1)*stride, because it is already
        // accounted for in the indices in 'ind_shuffled'
        Theiler_embed_mask(y+j+lag, npts, my,  1, stride, ind_shuffled, x_new,                            sp.N_eff);
        Theiler_embed_mask(y+j,     npts, my, py, stride, ind_shuffled, x_new+my*sp.N_eff,                sp.N_eff);
        Theiler_embed_mask(z+j,     npts, mz, pz, stride, ind_shuffled, x_new+my*(py+1)*sp.N_eff,         sp.N_eff);
        Theiler_embed_mask(x+j,     npts, mx, px, stride, ind_shuffled, x_new+(my*(py+1)+mz*pz)*sp.N_eff, sp.N_eff);

        if (USE_PTHREAD>0)  // if we want multithreading
            nb_errors += compute_partial_MI_direct_ann_threads(x_new, sp.N_eff, my, mx*px, my*py+mz*pz, k, &te1, &te2, get_cores_number(GET_CORES_SELECTED));
        else                 // single threaded algorithms:
            nb_errors += compute_partial_MI_engine_ann        (x_new, sp.N_eff, my, mx*px, my*py+mz*pz, k, &te1, &te2);
        
        avg1 += te1;    var1 += te1*te1;
		avg2 += te2;	var2 += te2*te2;
		last_npts_eff += last_npts_eff_local;
        
        free(ind_epoch);
        free(ind_shuffled);
        free_perm(perm_pts);
    } // loop on j
 
	avg1 /= sp.N_real;  var1 /= sp.N_real;  var1 -= avg1*avg1;
	avg2 /= sp.N_real;	var2 /= sp.N_real;	var2 -= avg2*avg2;
	*T1  = avg1;        last_std  = sqrt(var1);
	*T2  = avg2;    	last_std2 = sqrt(var2);
    
    nb_errors_total += nb_errors;
	last_samp=sp;

	free(x_new);
	return(nb_errors);
} /* end of function "compute_partial_TE_ann_mask" */




/*************************************************************************************
 * to compute partial MI "directly" from combinations of KL estimators
 * 
 * for the definition, and the estimator, see the article from Frenzel, Pompe 
 *		 PRL 99, 204101 (2007)
 *		 "Partial Mutual Information for Coupling Analysis of Multivariate Time Series"
 *
 * I(X,Y|Z) = part of the MI I(X,Y) which is not in Z
 *
 * x      contains the first variable (mx dimensions)
 * y      contains the second variable (my dimensions)
 * z      contains the conditionning variable (mz dimensions)
 * mask   contains the mask to apply on all the data (defining epochs of interest)
 * nx     is the number of points in time	
 * dim[0] is the dimension of x (usually 1, but maybe more)
 * dim[1] is the dimension of y (usually 1, but maybe more)
 * dim[2] is the dimension of z (usually 1, but maybe more)
 * px     is the number of points to consider in the past of x (embedding in x)
 * py     is the number of points to consider in the past of y (embedding in y)
 * pz     is the number of points to consider in the past of z (embedding in z)
 * stride is the distance between 2 points in time, in the past
 * k      is the number of neighbors to consider
 *
 * This function uses a mask to define relevant epochs
 *
 * This is just a wrapper to the function "compute_partial_MI_direct_ann"
 *
 * 2013-06-18: fork from compute_partial_MI_nd_ann 
 *              and getting inspiration from compute_mutual_information_nd_ann_mask
 * 2023-12-01: bug correction (same as in MI)
 *************************************************************************************/
int compute_partial_MI_ann_mask(double *x, double *y, double *z, char *mask, int npts, int *dim, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)
{	register int i, j, n, pp,
           mx=dim[0], my=dim[1], mz=dim[2], px=dim[3+0], py=dim[3+1], pz=dim[3+2];
	double *x_new;
    double mi1, mi2, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;
    int     N_real_max=0, npts_good;	
    int     *ind_epoch=NULL;
    size_t  *ind_shuffled=NULL;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations, .type=-1};
	gsl_permutation *perm_pts;

    *I1=my_NAN; *I2=my_NAN;
    
	if (stride<1)   return(printf("[compute_partial_MI_ann_mask] : stride has to be equal or larger than 1.\n"));
	if ((px<1) || (py<1) || (pz<1))
                    return(printf("[compute_partial_MI_ann_mask] : embedding dimensions must be at least 1 !\n"));
    if ((mx<1) || (my<1) || (mz<1))
                    return(printf("[compute_partial_MI_ann_mask] : initial dimensions of data must be at least 1 !\n"));
    if (k<1)        return(printf("[compute_partial_MI_ann_mask] : k has to be equal or larger than 1.\n"));
    
    n	   = mx*px + my*py + mz*pz;    // total dimension of the new variable
	pp     = (px>py) ? px : py;        // who has the largest past ?
	pp     = (pp>pz) ? pp : pz;

    N_real_max = set_sampling_parameters_mask(mask, npts, pp, stride, 0, &sp, "compute_partial_MI_ann_mask");
    if (N_real_max<1)   return(printf("[compute_partial_MI_ann_mask] : aborting\n"));
    if (sp.N_eff < 2*k) return(printf("[compute_partial_MI_ann_mask] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
    
    x_new    = (double*)calloc(n*sp.N_eff, sizeof(double));
  
    nb_errors=0; last_npts_eff=0;
    for (j=0; j<sp.N_real; j++)         // loop over independant windows (independant !!! cf Theiler !!!)
    {   
        npts_good = analyze_mask_for_sampling(mask+j, npts-j, pp, stride, 0, sp.Theiler, 1, &ind_epoch); 
        if (npts_good<sp.N_eff) 
        {   printf("\tnpts_good = %d < N_eff =%d!!!\n", npts_good, sp.N_eff);
            return(-1);
        }      
        perm_pts  = create_unity_perm(npts_good); shuffle_perm(perm_pts); // for random sampling
        ind_shuffled = (size_t*)calloc(npts_good, sizeof(size_t));
        for (i=0; i<npts_good; i++) ind_shuffled[i] = ind_epoch[perm_pts->data[i]]; // OK

        // note: in the calls below, the shift must not include (p-1)*stride, because it is already
        // accounted for in the indices in 'ind_shuffled'
        Theiler_embed_mask(x+j, npts, mx, px, stride, ind_shuffled, x_new,                        sp.N_eff);
        Theiler_embed_mask(z+j, npts, mz, pz, stride, ind_shuffled, x_new+(mx*px      )*sp.N_eff, sp.N_eff);
        Theiler_embed_mask(y+j, npts, my, py, stride, ind_shuffled, x_new+(mx*px+mz*pz)*sp.N_eff, sp.N_eff);
/*
    
    //        n_sets_total += nx_new;
        x_new  = (double*)calloc(n*nx_new, sizeof(double));
        for (i=0; i<nx_new; i++)        // loop on time over acceptable points for the current window
        {   for (d=0; d<mx; d++)        // loop over dimensions in x (future point)
            for (l=0; l<px; l++)        // loop over embedding in x
                x_new[i + (              + d + l*mx)*nx_new] = x[ind_epoch[i] + d*npts + shift - stride*l];
            for (d=0; d<mz; d++)        // loop over existing dimensions in z
            for (l=0; l<pz; l++)        // loop over embedding in z
                x_new[i + (mx*px         + d + l*mz)*nx_new] = z[ind_epoch[i] + d*npts + shift - stride*l];
            for (d=0; d<my; d++)        // loop over dimensions in y
            for (l=0; l<py; l++)        // loop over embedding  in y
                x_new[i + (mx*px + mz*pz + d + l*my)*nx_new] = y[ind_epoch[i] + d*npts + shift - stride*l];
        }
*/
        if (USE_PTHREAD>0) // if we want multithreading
            nb_errors += compute_partial_MI_direct_ann_threads(x_new, sp.N_eff, mx*px, my*py, mz*pz, k, &mi1, &mi2, get_cores_number(GET_CORES_SELECTED)); //PMI
        else
            nb_errors += compute_partial_MI_engine_ann(x_new, sp.N_eff, mx*px, my*py, mz*pz, k, &mi1, &mi2); //PMI
        avg1 += mi1;    var1 += mi1*mi1;
		avg2 += mi2;	var2 += mi2*mi2;
		last_npts_eff += last_npts_eff_local;
        
        free(ind_epoch);
        free(ind_shuffled);
        free_perm(perm_pts);
	}
    
	avg1 /= sp.N_real;  var1 /= sp.N_real;  var1 -= avg1*avg1;
	avg2 /= sp.N_real;	var2 /= sp.N_real;	var2 -= avg2*avg2;
	*I1  = avg1;        last_std  = sqrt(var1);
	*I2  = avg2;    	last_std2 = sqrt(var2);

    nb_errors_total += nb_errors;
	last_samp=sp;

	free(x_new);
	return(nb_errors);
} /* end of function "compute_partial_MI_nd_ann_mask" */




/*************************************************************************************
 * to compute Directed Information "directly"
 * 
 * We use the formula (3.6) from Kramer's thesis page 27
 * cf also VIII p 162
 *
 * DI(X->Y)   (same as TE)
 *
 * version with masking (epochs)
 *
 * x      contains the first variable (multidimensional)
 * y      contains the second variable (multidimensional)
 * mask   contains the mask to apply on data
 * npts   is the number of points in time	
 * N      is the number of points to consider in time (N>=1) ("embedding")
 * stride is the distance between 2 points in time, in the past
 * k      is the number of neighbors to consider
 *
 * This function uses the function "compute_partial_MI_direct_ann"
 *
 * 2012-03-10 : first version (ann, with kd-tree)
 * 2012-05-06 : added Theiler correction
 * 2013-03-06 : now operate on multidimensional data x and y
 * 2013-06-21 : new version with mask
 * 2019-01-24 : note only: to-do: check variable malloc size when d is varied
 *              and add a test on nx_new
 * 2020-02-28 : nb_errors and last_npts_eff not updated yet : to do!
 * 2020-02-28 : idea: perform a large embedding first, and then loop on dimension : to do?
 * 2022-06-10 : done + new samplings, untested yet
 * 2023-02-23 : now DI(X->Y) (1st -> 2nd argument) instead of the opposite
 *************************************************************************************/
int compute_directed_information_ann_mask(double *x, double *y, char *mask, int npts, int mx, int my, int N, int stride, 
                            int Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)
{	register int i, j, n;
	double *x_new, tmp1, tmp2;
	double *di1, *di2, *var1, *var2;      // 2022-06-10 : we'll keep a trace of all orders<=N of DI
    int     N_real_max=0, npts_good;	
    int     *ind_epoch=NULL;
    size_t  *ind_shuffled=NULL;
    samp_param  sp = { .Theiler=Theiler, .N_eff=N_eff, .N_real=N_realizations, .type=-1};
	gsl_permutation *perm_real, *perm_pts;

    *I1=my_NAN; *I2=my_NAN;

	if ((mx<1)||(my<1)) return(printf("[compute_directed_information_ann_mask] : nx and ny have to be equal or larger than 1.\n"));	
	if (N<1)            return(printf("[compute_directed_information_ann_mask] : N has to be equal or larger than 1.\n"));
	if (stride<1)       return(printf("[compute_directed_information_ann_mask] : stride has to be equal or larger than 1.\n"));
    if (k<1)            return(printf("[compute_directed_information_ann_mask] : k has to be equal or larger than 1.\n"));
    
    // additional checks and auto-adjustments of parameters, using max N:
    N_real_max = set_sampling_parameters_mask(mask, npts, N, stride, 0, &sp, "compute_directed_information_ann_mask");
    if (N_real_max<1)   return(printf("[compute_directed_information_ann_mask] : aborting !\n"));
    if (sp.N_eff < 2*k) return(printf("[compute_directed_information_ann_mask] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));

    di1    = (double*)calloc(N, sizeof(double));    var1 = (double*)calloc(N, sizeof(double));
    di2    = (double*)calloc(N, sizeof(double));    var2 = (double*)calloc(N, sizeof(double));
	x_new  = (double*)calloc((mx+my)*N*sp.N_eff, sizeof(double));
	
	perm_real = create_unity_perm(N_real_max); if (sp.type>=3) shuffle_perm(perm_real);
    
    nb_errors=0; last_npts_eff=0;       // global variables
    for (j=0; j<sp.N_real; j++)         // loop over realizations
    {               
        npts_good = analyze_mask_for_sampling(mask+perm_real->data[j], npts-perm_real->data[j], N, stride, 0, sp.Theiler, 1, &ind_epoch); 
        if (npts_good<sp.N_eff)     return(printf("\tnpts_good = %d < N_eff =%d!!!\n", npts_good, sp.N_eff));
            
        perm_pts  = create_unity_perm(npts_good); shuffle_perm(perm_pts); // for random sampling
        ind_shuffled = (size_t*)calloc(npts_good, sizeof(size_t));
        for (i=0; i<npts_good; i++) ind_shuffled[i] = ind_epoch[perm_pts->data[i]]; // OK

        for (n=1; n<=N; n++)            // loop on time-embedding, for a fixed realization since 2022-06-10
        {   // note: in the calls below, the shift must not include (p-1)*stride, because it is already
            // accounted for in the indices in 'ind_shuffled'
            Theiler_embed_mask(y+perm_real->data[j], npts, my, n, stride, ind_shuffled, x_new,               sp.N_eff);
            Theiler_embed_mask(x+perm_real->data[j], npts, mx, n, stride, ind_shuffled, x_new+my*n*sp.N_eff, sp.N_eff);

            if (n==1) // first term of the sum is regular MI (DI = MI if N==1): 
            {   if (USE_PTHREAD>0) nb_errors += compute_mutual_information_2xnd_ann_threads(x_new, sp.N_eff, 
                                                        my*n, mx*n, k, &tmp1, &tmp2, get_cores_number(GET_CORES_SELECTED));
                else               nb_errors += compute_mutual_information_direct_ann(x_new, sp.N_eff, 
                                                        my*n, mx*n, k, &tmp1, &tmp2);
                di1[0] += tmp1; var1[0] += tmp1*tmp1;        // 2023-02-24: replaced = by +=
                di2[0] += tmp2; var2[0] += tmp2*tmp2;
            }
            else
            {   if (USE_PTHREAD>0) nb_errors += compute_partial_MI_direct_ann_threads(x_new, sp.N_eff, 
                                                        my, mx*n, my*(n-1), k, &tmp1, &tmp2, get_cores_number(GET_CORES_SELECTED));
                else               nb_errors += compute_partial_MI_engine_ann(x_new, sp.N_eff, 
                                                        my, mx*n, my*(n-1), k, &tmp1, &tmp2);
                di1[n-1] += tmp1; var1[n-1] += tmp1*tmp1;
                di2[n-1] += tmp2; var2[n-1] += tmp2*tmp2;
            }
        } // end of loop over n
        last_npts_eff += last_npts_eff_local;
        
        free(ind_epoch);
        free(ind_shuffled);
        free_perm(perm_pts);
    } // end of loop over realizations (loop over j)

    for (n=0; n<N; n++)
    {	di1[n] /= sp.N_real;  var1[n] /= sp.N_real;  var1[n] -= di1[n]*di1[n];
        di2[n] /= sp.N_real;  var2[n] /= sp.N_real;  var2[n] -= di2[n]*di2[n];
    }

    // as of 2022-06-10, we only return the DI at order N:
    for (n=1; n<N; n++)
    {	di1[n] += di1[n-1];    var1[n] += var1[n-1];   
        di2[n] += di2[n-1];    var2[n] += var2[n-1];
    }
    *I1=di1[N-1];    last_std  = sqrt(var1[N-1]);
    *I2=di2[N-1];    last_std2 = sqrt(var2[N-1]);

    last_samp=sp;
    free(x_new);    
    free(di1); free(di2); free(var1); free(var2);
	return(nb_errors);
} /* end of function "compute_directed_information_ann_mask" */


