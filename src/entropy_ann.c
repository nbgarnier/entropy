/*
 *  entropy_ann.c
 *  
 *  to compute entropies with k-nn algorithms
 *  using ANN library : http://www.cs.umd.edu/~mount/ANN/
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2012-02-28 : fork from entropy_nns.c; include and use of ANN library
 *  2012-05-03 : Theiler correction properly implemented
 *               the behavior of the code with respect to the Theiler correction 
 *               is given by a #define USE_THEILER (2022-05: now deprecated)
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
 *  2021-12-21 : some functions moved to a new file "entropy_ann_combinations.c"
 *  2022-04-14 : fork from "entropy_ann.c" for new samplings of realizations (_N versions)
 *  2022-12-07 : changes in samplings.c + new extensive tests performed: legacy Theiler is operating well
 *  2023-11-28 : renamed "entropy_ann_N.c" into "entropy_ann.c", and all functions accordingly
 */

#include <math.h>                   // for fabs and others
#include <string.h>

#include "library_commons.h"        // for definitions of nb_errors, and stds
#include "library_matlab.h"         // compilation for Matlab
#include "ANN_wrapper.h"            // for ANN library functions (in C++)
#include "nns_count.h"              // NBG counting functions (2019-01-23)
#include "samplings.h"              // created 2021-12-17 (to factor some operations)
#include "math_tools.h"
#include "entropy_ann.h"
#include "entropy_ann_single_entropy.h"  // for the engine function to compute Shannon entropy
#include "entropy_ann_single_RE.h"       // for the engine function to compute RE
#include "entropy_ann_single_MI.h"       // for the engine function to compute MI with either NG or ANN counting
#include "entropy_ann_single_PMI.h"      // for the engine function to compute PMI with either NG or ANN counting
#include "entropy_ann_threads.h"         // for some global variables, and of course some functions
#include "entropy_ann_threads_entropy.h" // for the special threaded-function to compute Shannon entropy
#include "entropy_ann_threads_RE.h"      // for the special threaded-function to compute relative entropy
#include "entropy_ann_threads_MI.h"      // for the special threaded-function to compute MI
#include "entropy_ann_threads_PMI.h"     // for the special threaded-function to compute PMI


#define noDEBUG	    // for debug information, replace "noDEBUG" by "DEBUG"
#define noDEBUG_EXPORT
#define LOOK 17 	// for debug also (of which point(s) will we save the data ?)


#define noJAYNES      // 2024-03-05: Jaynes proposal for continous entropy

#ifdef JAYNES
#include <gsl/gsl_statistics_double.h>  // for std of intermediate (pre-processed) data
#endif

// constant to be used globally (including in other source .c files, and cython)
const int k_default = 5;

// choice of the Kraskov et al. algorithm for MIs:
// default = algo 1 from Kraskov, and legacy counting, and optimized masks
int MI_algo = MI_ALGO_1 | COUNTING_NG; // | MASK_OPTIMIZED;


#ifdef DEBUG
/* 2018-04-11/3: test of memory alignment: */
void debug_trace(char *text, double *x, int npts, int m, int p, int stride, int k)
{    register int i, j;
#define NPTS_T 9
     printf("%s - npts=%d, m=%d, p=%d, stride=%d, k=%d\n", text, npts, m, p, stride, k);
     for (j=0; j<m; j++)
     {    printf("  vector %d: [ ", j);
          for (i=0; i<(NPTS_T>npts?npts:NPTS_T); i++) printf("%f ",x[i+j*npts]);
          printf("] along time\n");
     }
     return;
}
#endif



/****************************************************************************************/
/* computes Shannon entropy, using nearest neighbor statistics (Grassberger 2004)	    */
/*                                                                                      */
/* this version is for m-dimentional systems, with eventually some stride/embedding	    */
/* here, N_eff is imposed, as is tau_Theiler                                            */
/*																			            */
/* x      contains all the data, which is of size nx in time					    	*/
/* nx     is the number of points in time											    */
/* m	  is the (initial) dimensionality of x								            */
/* p	  indicates how many points to take in time (in the past) (embedding)           */
/* tau    indicates the time lag between 2 consecutive points to be considered in time	*/
/* tau_Theiler    : mimimal stride between two sets of points                           */
/* N_eff          : number of points to use (ie, to consider for statistics)            */
/* N_realizations : number of realizations to consider (expected independant)           */
/* k      nb of neighbors to be considered										        */
/*																			            */
/* data is ordered like this :													        */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)				*/
/*																			            */
/* this function is a  wrapper to the functions :									    */
/*	- compute_entropy_nd_ann                                                            */
/*																			            */
/* 2011-11-15 : first version													        */
/* 2012-05-05 : added Theiler correction                                                */
/* 2012-06-01 : Theiler correction should also work with p=1 (stride>1) : new test      */
/* 2020-02-26 : added output of standard deviation                                      */
/* 2021-12-08 : using pthread                                                           */
/* 2021-12-17 : new function for embedding                                              */
/* 2022-04-14 : function "compute_entropy_ann_N", forked from "compute_entropy_ann"     */
/* 2023-11-28 : renamed "compute_entropy_ann" (old one is now "_legacy")                */
/****************************************************************************************/
int compute_entropy_ann(double *x, int npts, int m, int p, int tau, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *S)
{	register int j;
#ifdef JAYNES
    register int d; 
    double x_new_std[n];    // stack allocation (should be faster)
#endif
	int    n=m*p; // total dimensionality
	double *x_new, S_tmp, avg=0.0, var=0.0;
	int    N_real_max=0;
	samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations};
	gsl_permutation *perm_real, *perm_pts;
	
#ifdef DEBUG
    debug_trace("[compute_entropy_ann] signal x", x, npts, m, p, tau, k);
#endif
    *S = my_NAN;    // default returned value

    if ((m<1) || (p<1))     return(printf("[compute_entropy_ann] : m and p must be at least 1 !\n"));
    if ((tau<1))            return(printf("[compute_entropy_ann] : tau must be at least 1 !\n"));
    if ((k<1))              return(printf("[compute_entropy_ann] : k must be at least 1 !\n"));
	
    // additional checks and auto-adjustments of parameters:
    N_real_max = set_sampling_parameters(npts, p, tau, &sp, "compute_entropy_ann_N");

    if (N_real_max<1)       return(printf("[compute_entropy_ann] : aborting !\n"));
    if (sp.N_eff < 2*k)     return(printf("[compute_entropy_ann] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
    
    x_new  = (double*)calloc(n*sp.N_eff, sizeof(double));

    perm_real = create_unity_perm(N_real_max);  if (sp.type>=3) shuffle_perm(perm_real);
    perm_pts = create_unity_perm(sp.N_eff_max);     // for random sampling

    nb_errors=0; last_npts_eff=0;    
    for (j=0; j<sp.N_real; j++)   // loop over "independant" windows
    {   if (sp.type>=3) shuffle_perm(perm_pts);
    
        Theiler_embed(x+(tau*(p-1)+perm_real->data[j]), npts, m, p, tau, sp.Theiler, perm_pts->data, x_new, sp.N_eff);
#ifdef DEBUG
        debug_trace("[compute_entropy_ann_legacy] signal x_new", x_new, sp.N_eff, n, 1, tau, k);
#endif

#ifdef JAYNES // unfinished
        // 2024-03-05: std in each dimension for Jaynes correction:
        for (d=0; d<n; d++)
        {   x_new_std[d]  = gsl_stats_sd(x_new+d*N_eff, 1, sp.N_eff);
        }
        // 2024-03-05: compute Jaynes correction for all dimensions:
        // this may require the "real" std, ie, the one over all the signal, not just over the present realization
#endif

        if (USE_PTHREAD>0) S_tmp = compute_entropy_nd_ann_threads(x_new, N_eff, n, k, get_cores_number(GET_CORES_SELECTED));
        else               S_tmp = compute_entropy_nd_ann        (x_new, N_eff, n, k);
            
        avg  += S_tmp;
        var  += S_tmp*S_tmp;
        nb_errors += nb_errors_local; // each call to "compute_entropy_nd_ann" gives a new value of nb_errors_local
        last_npts_eff += last_npts_eff_local; // each call to "compute_entropy_nd_ann" uses an effective nb of points
    }
    avg /= sp.N_real;
    var /= sp.N_real;  var -= avg*avg;
    
    *S = avg;
    last_std = sqrt(var);
    
    nb_errors_total += nb_errors;
    last_samp=sp;

    free(x_new);     
    free_perm(perm_real);    free_perm(perm_pts);
	return(nb_errors);
} /* end of function "compute_entropy_ann" **********************************************/



/****************************************************************************************
 * computes relative Shannon entropy H(x||y), using nearest neighbor statistics         *
 * this is derived from: Leonenko et al, Annals of Statistics 36 (5) pp2153–2182 (2008) *
 *                                                                                      *
 * this version computes information redundency of 2 variables x and y                  *
 * of initial dimensions mx and my, after embedding of px points in x and py in y,      *
 * the final system is (mx*px + my*py)-dimensional                                      *
 *                                                                                      *
 * x    contains the reference dataset x, distributed from f(x). x is of size n*mx      *
 * nx   is the number of observations of x (nb of points in time)                       *
 * y    contains the dataset y, distributed from g(y). y is of size n*ny                *
 * ny   is the number of observations of y (nb of points in time)                       *
 * mx   is the nb of dimension of x before embedding                                    *
 * my   is the nb of dimension of y before embedding                                    *
 * px   indicates how many points to take in the past of x (embedding)                  *
 * py   indicates how many points to take in the past of y (embedding)                  *
 * stride is the time lag between 2 consecutive points in time when embedding           *
 *                                                                                      *
 * this function is a wrapper to the functions :                                        *
 *     - compute_relative_entropy_2xnd_ann                                              *
 *     - compute_relative_entropy_2xnd_ann_threads                                      *
 *                                                                                      *
 * 2017-11-29 : forked from "compute_mutual_information_nd_ann"                         *
 * 2020-02-26 : added output of standard deviation                                      *
 * 2021-12-14 : added threaded version                                                  *
 * 2022-04-15 : unfinished conversion: to do: check N_real_max in x nd y and adapt!     *
 * 2023-02-20 : little bugs correction                                                  *
 * 2023-11-28 : renamed from "compute_relative_entropy_ann_N" without the "_N" now      *
*****************************************************************************************/
int compute_relative_entropy_ann(double *x, int nx, double *y, int ny, int mx, int my, int px, int py, int tau, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *H)
{   double  *x_new, *y_new;
    double  avg=0.0, var=0.0, tmp;
    int     N_real_tot=0;
    register int j;   
    int     n_new=mx*px;
    samp_param  sp_x = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations},
                sp_y = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations};
	gsl_permutation *perm_real_x, *perm_real_y;
	gsl_permutation *perm_pts_x,  *perm_pts_y;

    *H=my_NAN;

    if ((mx<1) || (my<1)) return(printf("[compute_relative_entropy_ann] : mx and my must be at least 1 !\n"));
    if ((px<1) || (py<1)) return(printf("[compute_relative_entropy_ann] : px and py must be at least 1 !\n"));
    if ((tau<1))          return(printf("[compute_relative_entropy_ann] : tau must be at least 1 !\n"));
    if (k<1)              return(printf("[compute_relative_entropy_ann] : k has to be equal or larger than 1.\n"));
    if (mx*px != my*py)   return(printf("[compute_relative_entropy_ann] : resulting pdfs do not have the same dimensions!\n"));
         
    // additional checks and auto-adjustments of parameters:
    N_real_tot = set_sampling_parameters(nx, px, tau, &sp_x, "compute_relative_entropy_ann_N x");
    if (N_real_tot<1)     return(printf("[compute_relative_entropy_ann] : bad sampling in x, aborting !\n"));
    N_real_tot = set_sampling_parameters(ny, py, tau, &sp_y, "compute_relative_entropy_ann_N y");
    if (N_real_tot<1)     return(printf("[compute_relative_entropy_ann] : bad sampling in y, aborting !\n"));


    if (sp_x.N_eff < 2*k) return(printf("[compute_relative_entropy_ann x] : N_eff=%d is too small compared to k=%d)\n", sp_x.N_eff, k));
    if (sp_y.N_eff < 2*k) return(printf("[compute_relative_entropy_ann y] : N_eff=%d is too small compared to k=%d)\n", sp_y.N_eff, k));
    N_real_tot = (sp_x.N_real>sp_y.N_real) ? sp_y.N_real :sp_x.N_real;  // 2022-05-13: a convention for now...
//    N_eff_tot  = (sp_x.N_eff>sp_y.N_eff) ? sp_y.N_eff :sp_x.N_eff;      // 2022-05-13: a convention for now...
        
    x_new     = (double*)calloc(n_new*sp_x.N_eff, sizeof(double)); // embedded x
    y_new     = (double*)calloc(n_new*sp_y.N_eff, sizeof(double)); // embedded y
    
    // "generate" independant windows and samplings:
    perm_real_x = create_unity_perm(sp_x.N_real_max); if (sp_x.type>=3) shuffle_perm(perm_real_x);
    perm_real_y = create_unity_perm(sp_y.N_real_max); if (sp_y.type>=3) shuffle_perm(perm_real_y);
    perm_pts_x  = create_unity_perm(sp_x.N_eff_max);
    perm_pts_y  = create_unity_perm(sp_y.N_eff_max);
    
    nb_errors=0; last_npts_eff=0;  // global variables
    for (j=0; j<N_real_tot; j++)    // loop over independant windows
    {   if (sp_x.type>=3) { shuffle_perm(perm_pts_x); shuffle_perm(perm_pts_y); }
    
        Theiler_embed(x+(tau*(px-1)+perm_real_x->data[j]), nx, mx, px, tau, sp_x.Theiler, perm_pts_x->data, x_new, sp_x.N_eff);
        Theiler_embed(y+(tau*(py-1)+perm_real_y->data[j]), ny, my, py, tau, sp_y.Theiler, perm_pts_y->data, y_new, sp_y.N_eff);        

        if (USE_PTHREAD>0) // if we want multithreading
            tmp = compute_relative_entropy_2xnd_ann_threads(x_new, sp_x.N_eff, y_new, sp_y.N_eff, n_new, k,
                                get_cores_number(GET_CORES_SELECTED));            
        else
            tmp = compute_relative_entropy_2xnd_ann        (x_new, sp_x.N_eff, y_new, sp_y.N_eff, n_new, k);     
            
        avg += tmp;
        var += tmp*tmp;
        nb_errors += nb_errors_local;
        last_npts_eff += last_npts_eff_local; // each call to "compute_entropy_nd_ann" uses an effective nb of points
     }
     avg /=N_real_tot;
     var /=N_real_tot;  var -= avg*avg;
     
     *H  = avg;
     last_std = sqrt(var);
     
     nb_errors_total += nb_errors;
     last_samp=sp_x;
     
     free(x_new);               free(y_new);
     free_perm(perm_real_x);    free_perm(perm_pts_x);
     free_perm(perm_real_y);    free_perm(perm_pts_y);
     return(nb_errors);
} /* end of function "compute_relative_entropy_ann" *************************************/



/************************************************************************************/
/* computes mutual information, using nearest neighbor statistics                   */
/* this is an application of Grassberger PRE 69 066138 (2004)				        */
/*																			        */
/* this version computes information redundency of 2 variables x and y              */
/* of initial dimensions mx and my, after embedding of px points in x and py in y   */
/* final system is (mx*px + my*py)-dimentional                                      */
/*																			        */
/* x,y  contains all the datasets, which is of size nx	in time						*/
/* nx   is the number of points in time											    */
/* mx   is the nb of dimension of x before embedding                                */
/* my   is the nb of dimension of y before embedding                                */
/* px   indicates how many points to take in the past of x	(embedding)				*/
/* py   indicates how many points to take in the past of y 	(embedding)             */
/* stride is the time lag between 2 consecutive points in time when embedding		*/
/*																			        */
/* this function is a wrapper to the function :				     					*/
/*  - compute_mutual_information_direct_ann                                         */
/*	- compute_mutual_information_2xnd_ann_threads     							    */
/*																			        */
/* 2013-02-20 : first draft version, from "compute_mutual_information_ann"          */
/* 2013-02-22 : bug correction                                                      */
/* 2019-01-29 : added new (low-level) function using another counting (with ANN)    */
/* 2020-02-26 : added output of standard deviation                                  */
/* 2023-11-28 : renamed from "compute_mutual_information_ann_N" without the "_N"    */
/************************************************************************************/
int compute_mutual_information_ann(double *x, double *y, int nx, int mx, int my, int px, int py, int tau, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)
{	double *x_new;
	double mi1=0.0, mi2=0.0, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;
	register int j;	
	int     n = mx*px+my*py;
	int     pp = (px>py) ? px : py;	     // largest past;
    int     N_real_max=0;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations};
	gsl_permutation *perm_real, *perm_pts;

    *I1=my_NAN; *I2=my_NAN;

#ifdef DEBUG
     debug_trace("[compute_mutual_information_ann] signal x", x, nx, mx, px, tau, k);
     debug_trace("[compute_mutual_information_ann] signal y", y, nx, my, py, tau, k);
#endif
     
	if ((mx<1) || (my<1)) return(printf("[compute_mutual_information_ann] : mx and my must be at least 1 !\n"));
	if ((px<1) || (py<1)) return(printf("[compute_mutual_information_ann] : px and py must be at least 1 !\n"));
	if ((tau<1))          return(printf("[compute_mutual_information_ann] : tau must be at least 1 !\n"));
    if (k<1)              return(printf("[compute_mutual_information_ann] : k has to be at least 1.\n"));
    
    // additional checks and auto-adjustments of parameters:
    N_real_max = set_sampling_parameters(nx, pp, tau, &sp, "compute_mutual_information_ann");
    if (N_real_max<1)     return(printf("[compute_mutual_information_ann] : aborting !\n"));
    if (sp.N_eff < 2*k)   return(printf("[compute_mutual_information_ann] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
    
	x_new = (double*)calloc(n*sp.N_eff, sizeof(double));    // we work with a unique dataset
	
    perm_real = create_unity_perm(N_real_max); if (sp.type>=3) shuffle_perm(perm_real);
    perm_pts  = create_unity_perm(sp.N_eff_max);            // for random sampling

    nb_errors=0; last_npts_eff=0;
	for (j=0; j<sp.N_real; j++)  // loop over independant windows
	{	if (sp.type>=3) shuffle_perm(perm_pts);
	
        Theiler_embed(x+(tau*(pp-1)+perm_real->data[j]), nx, mx, px, tau, sp.Theiler, perm_pts->data, x_new,                sp.N_eff);
        Theiler_embed(y+(tau*(pp-1)+perm_real->data[j]), nx, my, py, tau, sp.Theiler, perm_pts->data, x_new+mx*px*sp.N_eff, sp.N_eff);

#ifdef DEBUG
         debug_trace("[compute_mutual_information_ann] signal x_new", x_new, sp.N_eff, n, px, tau, k);
#endif
        
        if (USE_PTHREAD>0) nb_errors += compute_mutual_information_2xnd_ann_threads(x_new, sp.N_eff, 
                                mx*px, my*py, k, &mi1, &mi2, get_cores_number(GET_CORES_SELECTED));
        else               nb_errors += compute_mutual_information_direct_ann(x_new, sp.N_eff, mx*px, my*py, k, &mi1, &mi2);
       
		avg1 += mi1;    var1 += mi1*mi1;
        avg2 += mi2;	var2 += mi2*mi2;
		last_npts_eff += last_npts_eff_local;
	}
	avg1 /= sp.N_real;  var1 /= sp.N_real;  var1 -= avg1*avg1;
	avg2 /= sp.N_real;  var2 /= sp.N_real;  var2 -= avg2*avg2;
	
	*I1  = avg1;        last_std  = sqrt(var1);
	*I2  = avg2;		last_std2 = sqrt(var2);
	
	nb_errors_total += nb_errors;
	last_samp=sp;
	
	free(x_new);
    free_perm(perm_real);    free_perm(perm_pts);
	return(nb_errors);
} /* end of function "compute_mutual_information_ann_N" **********************************/



/*************************************************************************************
 * to compute partial MI "directly" from combinations of KL estimators
 * 
 * for the definition, and the estimator, see the article
 *       Frenzel, Pompe - PRL 99, 204101 (2007)
 *		 "Partial Mutual Information for Coupling Analysis of Multivariate Time Series"
 *
 * I(X,Y|Z) = part of the MI I(X,Y) which is not in Z
 *
 * x      contains the first variable (mx dimensions)
 * y      contains the second variable (my dimensions)
 * z      contains the conditionning variable (mz dimensions)
 * npts   is the number of points in time	
 * dim[0] is the dimension of x (usually 1, but maybe more)
 * dim[1] is the dimension of y (usually 1, but maybe more)
 * dim[2] is the dimension of z (usually 1, but maybe more)
 * dim[3] is the number of points to consider in the past of x (embedding in x)
 * dim[4] is the number of points to consider in the past of y (embedding in y)
 * dim[5] is the number of points to consider in the past of z (embedding in z)
 * stride is the distance between 2 points in time, in the past
 * k      is the number of neighbors to consider
 *
 * This is just a wrapper to the function "compute_partial_MI_direct_ann"
 *
 * Be carefull : (x,y,z) are re-arranged as (x,z,y) before calling "compute_partial_MI_direct_ann"
 * but (m,p,q) are not. (convention...)
 * This should no be a problem, because "compute_partial_MI_direct_ann" should not be called directly
 *
 * 2012-05-09 : fork from compute_partial_MI_ann (dim_z=1)
 * 2013-02-22 : bug correction (function was still operating in 1-d in z)
 * 2013-02-22 : change of parameters syntax : now using an int pointer for dimensions and embedding
 * 2020-02-26 : added output of standard deviation
 * 2021-12-14 : now using unified function for PMI (handling both NBG or ANN counting)
 * 2023-11-28 : renamed from "compute_partial_MI_ann_N" without the "_N" now 
 *************************************************************************************/
int compute_partial_MI_ann(double *x, double *y, double *z, int npts, int *dim, int tau, 
                            int tau_Theiler, int N_eff, int N_realizations, int k, double *I1, double *I2)
{	register int j, shift, pp, mx=dim[0], my=dim[1], mz=dim[2], px=dim[3+0], py=dim[3+1], pz=dim[3+2];
    register int n = mx*px + my*py + mz*pz; // total dimension of the new variable
	double *x_new;
    double mi1, mi2, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;
    int N_real_max=0;
    samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations};
	gsl_permutation *perm_real, *perm_pts;

    *I1=my_NAN; *I2=my_NAN;
    
	if ((px<1) || (py<1) || (pz<1)) return(printf("[compute_partial_MI_ann] : embedding dimensions must be at least 1 !\n"));
    if ((mx<1) || (my<1) || (mz<1)) return(printf("[compute_partial_MI_ann] : initial dimensions of data must be at least 1 !\n"));
    if (tau<1)                      return(printf("[compute_partial_MI_ann] : tau has to be equal or larger than 1.\n"));
    if (k<1)                        return(printf("[compute_partial_MI_ann] : k must be at least 1 !\n"));
          
	pp         = (px>py) ? px : py;     // who has the largest past ?
	pp         = (pp>pz) ? pp : pz;
    
    // checks and auto-adjustments of sampling parameters:
    N_real_max = set_sampling_parameters(npts, pp, tau, &sp, "compute_partial_MI_ann");
    if (N_real_max<1)     return(printf("[compute_partial_MI_ann] : aborting !\n"));
    if (sp.N_eff < 2*k)   return(printf("[compute_partial_MI_ann] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));

	x_new  = (double*)calloc(n*sp.N_eff, sizeof(double));
    if (x_new==NULL) return(printf("[compute_partial_MI_ann] : memory allocation problem !\n"
                                    "\tplease check parameters (especially dimensions, total is %d)\n", n));
    perm_real = create_unity_perm(N_real_max); if (sp.type>=3) shuffle_perm(perm_real);
    perm_pts = create_unity_perm(sp.N_eff_max);     // for random sampling

	shift  = (pp-1)*tau;            // present is shifted
	nb_errors=0; last_npts_eff=0;   // global variables
    for (j=0; j<sp.N_real; j++)     // loop over independant windows
	{   if (sp.type>=3) shuffle_perm(perm_pts);
	
        Theiler_embed(x+shift+perm_real->data[j], npts, mx, px, tau, sp.Theiler, perm_pts->data, x_new                         , sp.N_eff);
        Theiler_embed(z+shift+perm_real->data[j], npts, mz, pz, tau, sp.Theiler, perm_pts->data, x_new+(      (mx*px)*sp.N_eff), sp.N_eff);
        Theiler_embed(y+shift+perm_real->data[j], npts, my, py, tau, sp.Theiler, perm_pts->data, x_new+((mx*px+mz*pz)*sp.N_eff), sp.N_eff);

        if (USE_PTHREAD>0) // if we want multithreading
        {   nb_errors += compute_partial_MI_direct_ann_threads(
                        x_new, sp.N_eff, mx*px, my*py, mz*pz, k, &mi1, &mi2, get_cores_number(GET_CORES_SELECTED));
        }
        else                // single threaded algorithms:
        {   nb_errors += compute_partial_MI_engine_ann(
                        x_new, sp.N_eff, mx*px, my*py, mz*pz, k, &mi1, &mi2);
        }

        avg1 += mi1;    var1 += mi1*mi1;
		avg2 += mi2;	var2 += mi2*mi2;
		nb_errors += nb_errors_local;
		last_npts_eff += last_npts_eff_local;
	}
	avg1 /= sp.N_real;  var1 /= sp.N_real;
	avg2 /= sp.N_real;	var2 /= sp.N_real;
	var1 -= avg1*avg1;
	var2 -= avg2*avg2;

	*I1  = avg1;        last_std  = sqrt(var1);
	*I2  = avg2;    	last_std2 = sqrt(var2);
	
	nb_errors_total += nb_errors;
    last_samp=sp;
    
    free(x_new);
    free_perm(perm_real);    free_perm(perm_pts);
	return(nb_errors);
} /* end of function "compute_partial_MI_ann" */


