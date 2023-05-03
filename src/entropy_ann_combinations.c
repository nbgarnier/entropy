/*
 *  entropy_ann_combinations.c
 *  
 *  to compute (combinations of) entropies with k-nn algorithms
 *  using ANN library : http://www.cs.umd.edu/~mount/ANN/
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2012-02-28 : fork from entropy_nns.c; include and use of ANN library
 *  2012-05-03 : Theiler correction properly implemented
 *               the behavior of the code with respect to the Theiler correction 
 *               is given by a #define USE_THEILER, in file "entropy_nns.h" (2022-05: now deprecated)
 *  2012-05-09 : added new function for PMI with conditioning variable z in a n-d space.
 *  2012-09-04 : bug correction in "compute_partial_MI_direct_ann()" (pb if m>1)
 *  2013-06-20 : forked into entropy_ann_mask.c for masking + slight improvements on tests
 *  2017-11-29 : renamed "search_ANN" as "search_ANN_internal" for future extensions
 *  2019-01-21 : added test for distance==0 inside ANN_wrapper.c (returned value!=0 if problem)
 *               and rewritten all tests for (nb_errors>=npts)
 *  2019-01-22 : rewritten some ANN functions, and cleaned up a little (tests moved back here)
 *  2020-02-26 : new management of "nb_errors" (via global variables)
 *  2021-11-25 : first multithreading tests
 *  2021-12-08 : engine functions moved in separate files
 *  2021-12-17 : new functions for embedding or increments
 *  2021-12-21 : new file "entropy_ann_combinations.c" split from "entropy_ann.c"
 *  2022-05-10 : new samplings propagated in all functions
 *  2023-02-23 : TE(X->Y) is now correctly ordered (was: TE(Y->X)); same with DI
 */

#include <math.h>                   // for fabs and others
#include <string.h>
#include <gsl/gsl_statistics_double.h>  // for std of intermediate (pre-processed) data

#include "library_commons.h"        // for definitions of nb_errors, and stds
#include "library_matlab.h"         // compilation for Matlab
#include "ANN_wrapper.h"            // for ANN library functions (in C++)
#include "nns_count.h"              // NBG counting functions (2019-01-23)
#include "samplings.h"              // created 2021-12-17 (to factor some operations)
#include "math_tools.h"
#include "entropy_ann.h"
#include "entropy_ann_N.h"
#include "entropy_ann_threads.h"
#include "entropy_ann_combinations.h"
#include "entropy_ann_single_entropy.h"  // for the engine function to compute Shannon entropy
#include "entropy_ann_single_RE.h"       // for the engine function to compute RE
#include "entropy_ann_single_MI.h"       // for the engine function to compute MI with either NG or ANN counting
#include "entropy_ann_single_PMI.h"      // for the engine function to compute PMI with either NG or ANN counting
#include "entropy_ann_threads_entropy.h" // for the special threaded-function to compute Shannon entropy
#include "entropy_ann_threads_RE.h"      // for the special threaded-function to compute relative entropy
#include "entropy_ann_threads_MI.h"      // for the special threaded-function to compute MI
#include "entropy_ann_threads_PMI.h"     // for the special threaded-function to compute PMI

#define noDEBUG	    // for debug information, replace "noDEBUG" by "DEBUG"
#define noDEBUG_EXPORT
#define LOOK 17 	// for debug also (which point(s) to save the data of ?)
#define TEST_EMBED  // new samplings!
#define noTIMING

#ifdef TIMING
    #include "timings.h"
#else 
    #define tic()
    #define toc(x) 0.0
#endif



/****************************************************************************************/
/* computes Shannon entropy of increments, using nearest neighbor statistics    	    */
/*                                                                                      */
/* this version is for m-dimentional systems, using increments of eventually high order */
/*																			            */
/* x      contains all the data, which is of size nx in time					    	*/
/* nx     is the number of points in time											    */
/* m	  indicates the (initial) dimensionality of x								    */
/* p	  indicates the order of increments (usually 1)                                 */
/* stride is the time lag between 2 consecutive points to be considered in time		    */
/* k      nb of neighbors to be considered										        */
/*																			            */
/* data is ordered like this :													        */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)				*/
/*																			            */
/* this function is a  wrapper to the functions :									    */
/*	- compute_entropy_nd_ann                                                            */
/*																			            */
/* 2021-12-17 : new function                                                            */
/* 2022-01-08 : new increments type (1 for regular incr. or 2 for averaged)             */
/* 2022-05-14 : new samplings, tested OK on Modane data on 2022-06-01                   */
/****************************************************************************************/
int compute_entropy_increments_ann(double *x, int npts, int m, int px, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, int incr_type, double *S)
{	
    register int j,
                 p = px+1;  // by analogy with embedding
	int    n = m;           // total dimensionality (same as input data)
	int    N_real_max=0;
	double *x_new, S_tmp, avg=0.0, var=0.0;
	double x_new_std=0.;    // 2022-03-11: for the std of the pre-processed signal
	samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations};
	gsl_permutation *perm_real, *perm_pts;
	
#ifdef DEBUGp
    debug_trace("[compute_entropy_increments_ann] signal x", x, npts, m, p, stride, k);
#endif

#ifdef NAN // default returned value
    *S = NAN;
#endif

    if ((m<1))          return(printf("[compute_entropy_increments_ann] : m must be at least 1 !\n"));
    if ((px<0))         return(printf("[compute_entropy_increments_ann] : p must be at least 0 !\n"));
    if ((stride<1))     return(printf("[compute_entropy_increments_ann] : stride must be at least 1 !\n"));
    if ((k<1))          return(printf("[compute_entropy_increments_ann] : k must be at least 1 !\n"));

    // additional checks and auto-adjustments of parameters:
/*    N_real_max = set_sampling_parameters_old(npts, p, stride, &sp, "compute_entropy_increments_ann_N");
    printf("stride  : %d\tmax : %d (Theiler) old function:\n", stride, tau_Theiler);
    print_samp_param(sp); fflush(stdout);
    sp.Theiler=tau_Theiler; sp.N_eff=N_eff; sp.N_real=N_realizations;
*/
    N_real_max = set_sampling_parameters(npts, p, stride, &sp, "compute_entropy_increments_ann_N");
/*    printf("stride  : %d\tmax : %d (Theiler) new function:\n", stride, tau_Theiler);
    print_samp_param(sp); fflush(stdout);
    printf("\n");
*/

    if (N_real_max<1)   return(printf("[compute_entropy_increments_ann] : aborting ! (Theiler %d, N_eff %d, N_real %d)\n", 
                                sp.Theiler, sp.N_eff, sp.N_real));
    if (sp.N_eff < 2*k) return(printf("[compute_entropy_increments_ann] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
    
    x_new  = (double*)calloc(n*sp.N_eff, sizeof(double));

    perm_real = create_unity_perm(N_real_max); shuffle_perm(perm_real);     // for independant windows
    // 2022-05-23: note: shuffling is a time-consuming operation, so large N_real_max should be avoided (Theiler=-4)
    perm_pts  = create_unity_perm(sp.N_eff_max);                            // for random sampling
    
    nb_errors=0; last_npts_eff=0;
    data_std=0;  data_std_std=0;        // 2022-03-11, for std of the increments
    for (j=0; j<sp.N_real; j++)         // loop over independant windows
    {   if (sp.type>=3) shuffle_perm(perm_pts);
        
        if (incr_type==1) increments(x+(stride*(p-1)+perm_real->data[j]), npts, m, p, stride, sp.Theiler, perm_pts->data, x_new, sp.N_eff);
        else              incr_avg  (x+(stride*(p-1)+perm_real->data[j]), npts, m, p, stride, sp.Theiler, perm_pts->data, x_new, sp.N_eff);

        // 2022-03-11: std of the increments:
        x_new_std     = gsl_stats_sd(x_new, 1, sp.N_eff);
        data_std     += x_new_std;
        data_std_std += x_new_std*x_new_std;

        if (USE_PTHREAD>0) S_tmp = compute_entropy_nd_ann_threads(x_new, sp.N_eff, n, k, get_cores_number(GET_CORES_SELECTED));
        else               S_tmp = compute_entropy_nd_ann        (x_new, sp.N_eff, n, k);
        
        avg  += S_tmp;
        var  += S_tmp*S_tmp;
        nb_errors += nb_errors_local; 
        last_npts_eff += last_npts_eff_local; 
    }
    avg /= sp.N_real;
    var /= sp.N_real;  var -= avg*avg;
    data_std     /= sp.N_real; 
    data_std_std /= sp.N_real;  data_std_std -= data_std*data_std;
    
    *S = avg;
    last_std = sqrt(var);
    nb_errors_total += nb_errors;
    last_samp=sp;
    
    free(x_new); 
    free_perm(perm_real);    free_perm(perm_pts);
	return(nb_errors);
} /* end of function "compute_entropy_increments_ann" ***********************************/



/****************************************************************************************/
/* computes Shannon entropy rate of order p, using nearest neighbor statistics          */
/*                                                                                      */
/* this version is for m-dimentional systems, with eventually some stride/embedding     */
/*                                                                                      */
/* x      contains all the data, which is of size nx in time                            */
/* npts   is the number of points in time                                               */
/* m      indicates the (initial) dimensionality of x                                   */
/* p      indicates how many points to take in time (in the past) (embedding)           */
/* stride is the time lag between 2 consecutive points to be considered in time         */
/* k      nb of neighbors to be considered                                              */
/* method is the method to use : 0 for fraction, 1 for difference and 2 for H-MI        */
/*                                                                                      */
/* data is ordered like this :                                                          */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)               */
/*                                                                                      */
/* this function is a wrapper over existing functions:                                  */
/*    - compute_entropy_ann                                                             */
/*    - compute_mutual_information_ann                                                  */
/* which are already managing threads by thermselves                                    */
/*                                                                                      */
/* 2019-01-26 : 3 methods                                                               */
/* 2020-02-23 : bug corrected in method=2 (MI)                                          */
/* 2020-02-26 : does not output the standard deviation (yet?)                           */
/* 2021-12-15 : now using copies of data to be OK for non-stationary signals            */
/* 2022-05-14 : new samplings, tested OK on Modane data on 2022-06-01                   */
/****************************************************************************************/
int compute_entropy_rate_ann(double *x, int npts, int m, int p, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S)
{   register int i, d;
    double H_past=0.0, H=0.0, I1=0.0, I2=0.0;
    double *x_past, *x_now;
        
#ifdef NAN // default returned value
    *S = NAN;
#endif
    
    if ((method!=ENTROPY_RATE_FRACTION) && (method!=ENTROPY_RATE_DIFFERENCE) && (method!=ENTROPY_RATE_MI))
    {                   return(printf("[compute_entropy_rate_ann] : invalid method (should be 0,1 or 2)!\n")); }
    if ((m<1) || (p<1)) return(printf("[compute_entropy_rate_ann] : m and p must be at least 1 !\n"));
    if ((stride<1))     return(printf("[compute_entropy_rate_ann] : stride must be at least 1 !\n"));
    if ((k<1))          return(printf("[compute_entropy_rate_ann] : k must be at least 1 !\n"));
    
    if (method==ENTROPY_RATE_FRACTION)
    {   nb_errors = compute_entropy_ann_N(x, npts, m, p, stride, tau_Theiler, N_eff, N_realizations, k, &H);
        *S = H/p;
    }
    else if (method==ENTROPY_RATE_DIFFERENCE)
    {   x_past = calloc( (npts-stride)*m , sizeof(double));
        x_now  = x;
        
        for (d=0; d<m; d++)
        for (i=0; i<npts-stride; i++)
        {   x_past[i + d*(npts-stride)] = x[i+d*npts];
        }
        nb_errors = compute_entropy_ann_N(x_past, npts-stride, m, p,   stride, tau_Theiler, N_eff, N_realizations, k, &H_past); 
        nb_errors = compute_entropy_ann_N(x_now,  npts,        m, p+1, stride, tau_Theiler, N_eff, N_realizations, k, &H);
        *S = H-H_past;
        
        free(x_past);
    }
    else if (method==ENTROPY_RATE_MI)  // 2020-02-23: causal embedding is performed in MI function below, OK
    {   x_past = calloc( (npts-stride)*m , sizeof(double));
        x_now  = calloc( (npts-stride)*m , sizeof(double));
        
        for (d=0; d<m; d++)
        for (i=0; i<npts-stride; i++)
        {   x_past[i + d*(npts-stride)] = x[i        + d*npts];
            x_now [i + d*(npts-stride)] = x[i+stride + d*npts];
        }
        nb_errors = compute_entropy_ann_N           (x_now,         npts-stride, m,    1,    stride, tau_Theiler, N_eff, N_realizations, k, &H);
        nb_errors = compute_mutual_information_ann_N(x_now, x_past, npts-stride, m, m, 1, p, stride, tau_Theiler, N_eff, N_realizations, k, &I1, &I2); 
        *S = (MI_algo&MI_ALGO_1) ? H-I1 : H-I2;
        
        free(x_past); free(x_now);
    }
    
    // 2021-12-15: by default, std and nb_errors are obtained from the last function call, 
    // which is the most "multi-dimensional". As a cosnequence, both quantities are over-estimates
    // (especially the std which is supposed to be much smaller due to compensations)
    return(nb_errors);
} /* end of function "compute_entropy_rate_ann" *************************************/



/************************************************************************************/
/* computes regularity index, using nearest neighbor statistics                     */
/* this index is the difference H(increments)-h (entropy rate)                      */
/* this is an application of Entropy (2021)				                            */
/*																			        */
/* this version computes the MI between the signal and its "next" increment         */
/*																			        */
/* x    contains the dataset, which is of size npts	in time  						*/
/* npts is the number of points in time											    */
/* mx   is the nb of dimension of x before embedding                                */
/* px   indicates how many points to take in the past of x	(embedding)				*/
/* stride is the time lag between 2 consecutive points in time when embedding		*/
/*																			        */
/* this function is a wrapper to the function :				     					*/
/*	- compute_mutual_information_2xnd_ann (if using threads)					    */
/*  - compute_mutual_information_direct_ann (otherwise)                             */
/*																			        */
/* 2021-12-19 : new function                                                        */
/* 2022-05-26 : new samplings, tested OK on Modane data on 2022-06-01               */
/************************************************************************************/
int compute_regularity_index_ann(double *x, int npts, int mx, int px, int stride, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)
{	double *x_new;
	double mi1=0.0, mi2=0.0, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;
	register int j;
	int     incr_type=1, // 2022-05-26 : may be changed later
	        n, N_real_max=0;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations};
	gsl_permutation *perm_real, *perm_pts;

#ifdef NAN
    *I1=NAN; *I2=NAN;
#endif
     
	if ((mx<1))     return(printf("[compute_regularity_index_ann] : mx must be at least 1 !\n"));
	if ((px<1))     return(printf("[compute_regularity_index_ann] : px must be at least 1 !\n"));
	if ((px!=1))    return(printf("[compute_regularity_index_ann] : only px=1 is curently supported.\n"));
	if (stride<1)   return(printf("[compute_regularity_index_ann] : stride must be at least 1 !\n"));
    if (k<1)        return(printf("[compute_regularity_index_ann] : k must be at least 1.\n"));
    
	n = mx*(px+1);
	
	N_real_max = set_sampling_parameters(npts, px+1, stride, &sp, "compute_regularity_index_ann");
    if (N_real_max<1) return(printf("[compute_regularity_index_ann] : aborting !\n"));
    if (sp.N_eff<2*k) return(printf("[compute_regularity_index_ann] : not enough points (%d vs k=%d)\n", sp.N_eff, k));
	
    x_new     = (double*)calloc(n*sp.N_eff, sizeof(double));
    perm_real = create_unity_perm(N_real_max); if (sp.type>=3) shuffle_perm(perm_real);
    perm_pts  = create_unity_perm(sp.N_eff_max);         // for random sampling

    nb_errors=0; last_npts_eff=0;
	for (j=0; j<sp.N_real; j++)  // loop over independant windows
	{   if (sp.type>=3) shuffle_perm(perm_pts);
	
        Theiler_embed(x+(stride*(px  -1)+perm_real->data[j]), npts, mx, px, stride, sp.Theiler, perm_pts->data, x_new, sp.N_eff);
        if (incr_type==1) increments(x+(stride*(px+1-1)+perm_real->data[j]), npts, mx,  2, stride, sp.Theiler, perm_pts->data, x_new+mx*px*sp.N_eff, sp.N_eff);
        else              incr_avg  (x+(stride*(px+1-1)+perm_real->data[j]), npts, mx,  2, stride, sp.Theiler, perm_pts->data, x_new+mx*px*sp.N_eff, sp.N_eff);
        
        if (USE_PTHREAD>0) nb_errors += compute_mutual_information_2xnd_ann_threads(x_new, sp.N_eff, 
                                mx*px, mx, k, &mi1, &mi2, get_cores_number(GET_CORES_SELECTED));
        else               nb_errors += compute_mutual_information_direct_ann      (x_new, sp.N_eff, 
                                mx*px, mx, k, &mi1, &mi2);

		avg1 += mi1;    var1 += mi1*mi1;
        avg2 += mi2;	var2 += mi2*mi2;
		last_npts_eff += last_npts_eff_local;
	}
	avg1 /= sp.N_real;  var1 /= sp.N_real;  var1 -= avg1*avg1;
	avg2 /= sp.N_real;  var2 /= sp.N_real;	var2 -= avg2*avg2;
	*I1  = avg1;        last_std  = sqrt(var1);
	*I2  = avg2;		last_std2 = sqrt(var2);
	
    nb_errors_total += nb_errors;
	last_samp=sp;
	
	free(x_new);
	free_perm(perm_real);    free_perm(perm_pts);
	return(nb_errors);
} /* end of function "compute_regularity_index_ann" **********************************/



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
 * nx     is the number of points in time	 (same for x and y)
 * mx     is the number of dimensions of x
 * my     is the number of dimensions of y
 * px     is the number of points to consider in the past of x (embedding)
 * py     is the number of points to consider in the past of y (embedding)
 * stride is the distance between 2 points in time, in the past
 * lag    is the distance between present y_n, and future point y_(n+lag)
 * k      is the number of neighbors to consider
 *
 * This is just a wrapper to the function "compute_partial_MI_direct_ann"
 *
 * 2013-03-01 : fork from function "compute_transfer_entropy_ann"
 * 2020-02-26 : added output of standard deviation
 * 2021-12-14 : now using unified function for PMI (handling both NBG or ANN counting)
 * 2022-05-10 : now with new samplings (Theiler based on stride, not lag)
 * 2023-02-23 : now TE(X->Y) (1st -> 2nd argument) instead of the opposite 
 ****************************************************************************************/
int compute_transfer_entropy_ann(double *x, double *y, int nx, int mx, int my, int px, int py, int stride, int lag, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *T1, double *T2)
{	register int j, shift;
    register int nn	= my*(py+1)+mx*px;   // dimension of the new variable
    register int pp = (px>py) ? px : py; // who has the largest past ?
#ifndef TEST_EMBED
    register int i,l,d;
#endif 
	double *x_new;
	double te1=0.0, te2=0.0, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;
    int     N_real_max=0;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations};
	gsl_permutation *perm_real, *perm_pts;

#ifdef NAN
    *T1=NAN; *T2=NAN;
#endif

	if (lag<1)          return(printf("[compute_transfer_entropy_ann] : lag has to be equal or larger than 1.\n"));
	if (stride<1)       return(printf("[compute_transfer_entropy_ann] : stride must be at least 1 !\n"));
    if ((mx<1)||(my<1)) return(printf("[compute_transfer_entropy_ann] : mx and my must be larger than 1 !\n"));
    if ((px<1)||(py<1)) return(printf("[compute_transfer_entropy_ann] : px and py must be larger than 1 !\n"));
    if (k<1)            return(printf("[compute_transfer_entropy_ann] : k has to be equal or larger than 1.\n"));

	// additional checks and auto-adjustments of parameters:
    N_real_max = set_sampling_parameters(nx-lag, pp, stride, &sp, "compute_transfer_entropy_ann");
    if (N_real_max<1)     return(printf("[compute_transfer_entropy_ann] : aborting !\n"));
    if (sp.N_eff < 2*k)   return(printf("[compute_transfer_entropy_ann] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));

    x_new  = (double*)calloc(nn*sp.N_eff, sizeof(double));
	shift  = (pp-1)*stride;   // present is shifted
	
	perm_real = create_unity_perm(N_real_max); if (sp.type>=3) shuffle_perm(perm_real);
    perm_pts  = create_unity_perm(sp.N_eff_max);         // for random sampling
    
    nb_errors=0; last_npts_eff=0;
    for (j=0; j<sp.N_real; j++) // loop over independant windows
	{   if (sp.type>=3) shuffle_perm(perm_pts);
#ifdef TEST_EMBED
        Theiler_embed(y+shift+lag+perm_real->data[j], nx, my,  1, stride, sp.Theiler, perm_pts->data, x_new,                      sp.N_eff);
        Theiler_embed(y+shift+perm_real->data[j],     nx, my, py, stride, sp.Theiler, perm_pts->data, x_new+(      my *sp.N_eff), sp.N_eff);
        Theiler_embed(x+shift+perm_real->data[j],     nx, mx, px, stride, sp.Theiler, perm_pts->data, x_new+(my*(py+1)*sp.N_eff), sp.N_eff);
#else
	    for (i=0; i<nx_new; i++) // loop over points in 1 window
        {   {   for (l=0; l<mx; l++) // loop over dimensions in x
                x_new[i +                     l*nx_new ] = x[j + n_windows*i + shift + lag + l*nx];
            }
            for (d=0; d<px; d++) // loop over embedding in x
            {   for (l=0; l<mx; l++) // loop over dimensions in x
                x_new[i +      (mx*1 + d + l*px)*nx_new] = x[j + n_windows*i + shift - stride*d + l*nx];
            }
            for (d=0; d<py; d++) 
            {   for (l=0; l<my; l++)
                x_new[i + (mx*(px+1) + d + l*py)*nx_new] = y[j + n_windows*i + shift - stride*d + l*nx];
            }
        }
        // now we have a vector with all components of interest (see VIII p 148 ) 
#endif
        if (USE_PTHREAD>0) // if we want multithreading
            nb_errors += compute_partial_MI_direct_ann_threads(x_new, sp.N_eff, 
                my, mx*px, my*py, k, &te1, &te2, get_cores_number(GET_CORES_SELECTED));
        else                // single threaded algorithms:
            nb_errors += compute_partial_MI_engine_ann        (x_new, sp.N_eff, 
                my, mx*px, my*py, k, &te1, &te2);
            
        avg1 += te1;    var1 += te1*te1;
		avg2 += te2;	var2 += te2*te2;
		last_npts_eff += last_npts_eff_local;
	}
    if (nb_errors!=0) return(printf("[compute_transfer_entropy_nd_ann] error, %d points were problematic\n", nb_errors));

    avg1 /= sp.N_real;  var1 /= sp.N_real;
	avg2 /= sp.N_real;	var2 /= sp.N_real;
	var1 -= avg1*avg1;
	var2 -= avg2*avg2;
	*T1 = avg1;         last_std  = sqrt(var1);
	*T2 = avg2;         last_std2 = sqrt(var2);
	
    nb_errors_total += nb_errors;
    last_samp=sp;
    
    free(x_new);
    free_perm(perm_real);    free_perm(perm_pts);
	return(nb_errors);
} /* end of function "compute_transfer_entropy_ann" */



/*************************************************************************************
 * to compute Directed Information
 * 
 * We use the formula (3.6) from Kramer's thesis page 27
 * cf also VIII p 162
 *
 * DI(X->Y) : influence of X (1st argument) on Y (2nd argument)
 *
 * x      contains the first variable (multidimensional)
 * y      contains the second variable (multidimensional)
 * npts   is the number of points in time	
 * mx, my are the dimensionalities of x and y
 * N      is the number of points to consider in time (N>=2) ("embedding")
 * stride is the distance between 2 points in time, in the past
 * k      is the number of neighbors to consider
 *
 * This function uses the functions:
 *  - "compute_mutual_information_ann"
 *  - "compute_partial_MI_direct_ann"
 *
 * 2012-03-10 : first version (ann, with kd-tree)
 * 2012-05-06 : added Theiler correction
 * 2013-03-06 : now operate on multidimensional data x and y
 * 2020-02-26 : added output of standard deviation 
 * 2021-12-14 : now using unified function for PMI (handling both NBG or ANN counting)
 * 2022-06-10 : exchanged loops on embedding and on realizations, in order to use the same points
 * 2023-02-23 : now DI(X->Y) (1st -> 2nd argument) instead of the opposite 
 *************************************************************************************/
int compute_directed_information_ann(double *x, double *y, int npts, int mx, int my, int N, int stride, 
                            int Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)
{	register int j, shift, n;
	double *x_new, tmp1, tmp2;
	double *di1, *di2, *var1, *var2;      // 2022-06-10 : we'll keep a trace of all orders<=N of DI
	int     N_real_max=0;
    samp_param  sp = { .Theiler=Theiler, .N_eff=N_eff, .N_real=N_realizations};
	gsl_permutation *perm_real, *perm_pts;

#ifdef NAN
    *I1=NAN; *I2=NAN;
#endif
    
	if ((mx<1)||(my<1)) return(printf("[compute_directed_information_ann] : mx and my have to be equal or larger than 1.\n"));	
	if (N<1)            return(printf("[compute_directed_information_ann] : N has to be equal or larger than 1.\n"));	
	if (stride<1)       return(printf("[compute_directed_information_ann] : stride has to be equal or larger than 1.\n"));
    if (k<1)            return(printf("[compute_directed_information_ann] : k has to be equal or larger than 1.\n"));

    // additional checks and auto-adjustments of parameters, using max N:
    N_real_max = set_sampling_parameters(npts, N, stride, &sp, "compute_directed_information_ann");
    if (N_real_max<1)   return(printf("[compute_directed_information_ann] : aborting !\n"));
    if (sp.N_eff < 2*k) return(printf("[compute_directed_information_ann] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
    perm_real = create_unity_perm(N_real_max); if (sp.type>=3) shuffle_perm(perm_real);
    perm_pts  = create_unity_perm(sp.N_eff_max);     // for random sampling

    di1    = (double*)calloc(N, sizeof(double));    var1 = (double*)calloc(N, sizeof(double));
    di2    = (double*)calloc(N, sizeof(double));    var2 = (double*)calloc(N, sizeof(double));
	x_new  = (double*)calloc((mx+my)*N*sp.N_eff, sizeof(double));
	
    nb_errors=0; last_npts_eff=0;       // global variables
    for (j=0; j<sp.N_real; j++)         // loop over realizations
    {   if (sp.type>=3) shuffle_perm(perm_pts);
            
        for (n=1; n<=N; n++)            // loop on time-embedding, for a fixed realization since 2022-06-10
        {   shift  = (n-1)*stride;      // present is shifted

            Theiler_embed(y+shift+perm_real->data[j], npts, my, n, stride, sp.Theiler, perm_pts->data, x_new                , sp.N_eff);
            Theiler_embed(x+shift+perm_real->data[j], npts, mx, n, stride, sp.Theiler, perm_pts->data, x_new+(my*n*sp.N_eff), sp.N_eff);
            
            if (n==1)                   // first term of the sum is regular MI (DI = MI if N==1)
            {   if (USE_PTHREAD>0) nb_errors += compute_mutual_information_2xnd_ann_threads(x_new, sp.N_eff, 
                                                        my*n, mx*n, k, &tmp1, &tmp2, get_cores_number(GET_CORES_SELECTED));
                else               nb_errors += compute_mutual_information_direct_ann(x_new, sp.N_eff, 
                                                        my*n, mx*n, k, &tmp1, &tmp2);
                di1[0] += tmp1; var1[0] += tmp1*tmp1;
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
	
    nb_errors_total += nb_errors;
    last_samp=sp;
   
    free(di1); free(di2); free(var1); free(var2);
    
    free(x_new);
   	free_perm(perm_real);    free_perm(perm_pts);
	return(nb_errors);
} /* end of function "compute_directed_information_ann" */

