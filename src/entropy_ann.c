/*
 *  entropy_ann.c
 *  
 *  to compute entropies with k-nn algorithms
 *  using ANN library : http://www.cs.umd.edu/~mount/ANN/
 *
 *  Created by Nicolas Garnier on 2012/02/25.
 *  Copyright 2012-2019 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2012-02-28 : fork from entropy_nns.c; include and use of ANN library
 *  2012-05-03 : Theiler correction properly implemented
 *               the behavior of the code with respect to the Theiler correction 
 *               is given by the #define USE_THEILER, in file "entropy_nns.h" (2022-05: now deprecated)
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
#define TEST_EMBED // changed 2022-12-07 for tests


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
/*																			            */
/* x      contains all the data, which is of size nx in time					    	*/
/* nx     is the number of points in time											    */
/* m	     indicates the (initial) dimensionality of x								*/
/* p	     indicates how many points to take in time (in the past) (embedding)        */
/* stride is the time lag between 2 consecutive points to be considered in time		    */
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
/****************************************************************************************/
int compute_entropy_ann(double *x, int npts, int m, int p, int stride, int k, double *S)
{	register int j;
#ifndef TEST_EMBED
    register int i,l,d;
#endif    
	int    n, nx_new, n_windows;
	double *x_new, S_tmp, avg=0.0, var=0.0;
	
#ifdef DEBUGp
    debug_trace("[compute_entropy_ann] signal x", x, npts, m, p, stride, k);
#endif

#ifdef NAN // default returned value
    *S = NAN;
#endif

    if ((m<1) || (p<1)) return(printf("[compute_entropy_ann] : m and p must be at least 1 !\n"));
    if ((stride<1))     return(printf("[compute_entropy_ann] : stride must be at least 1 !\n"));
    if ((k<1))          return(printf("[compute_entropy_ann] : k must be at least 1 !\n"));
    
	n = m*p; // total dimensionality
	
    // Theiler method 
    nx_new    = (npts-npts%stride)/stride - (p-1); // size of a single dataset
    n_windows = stride; // there are n_window different datasets to work with (tau_Theiler)
	    
    if (nx_new < 2*k) return(printf("[compute_entropy_ann] : not enough points in data"
                                " (npts/stride=%d/%d => %d effective pts vs k=%d)\n", npts, stride, nx_new, k));
    
    nb_errors=0; last_npts_eff=0;
    x_new  = (double*)calloc(n*nx_new, sizeof(double));
    
    for (j=0; j<n_windows; j++)   // loop over independant windows
    {   
#ifdef TEST_EMBED
        Theiler_embed_old(x+(stride*(p-1)+j), npts, m, p, stride, stride, x_new, nx_new);
#else    
        for (i=0; i<nx_new; i++)  // loop on time in 1 window
        {   for (d=0; d<m; d++)   // loop on existing dimensions in x
            for (l=0; l<p; l++)   // loop on embedding
                 x_new[i + nx_new*( d + l*m )] = x[j + n_windows*i + d*npts + stride*(p-1-l)];
        }
#endif        
#ifdef DEBUGp
        debug_trace("[compute_entropy_ann] signal x_new", x_new, nx_new, n, 1, stride, k);
#endif

        if (USE_PTHREAD>0) 
        {   S_tmp = compute_entropy_nd_ann_threads(x_new, nx_new, n, k,
                get_cores_number(GET_CORES_SELECTED));
        }
        else
            S_tmp = compute_entropy_nd_ann        (x_new, nx_new, n, k);
            
        avg  += S_tmp;
        var  += S_tmp*S_tmp;
        nb_errors += nb_errors_local; // each call to "compute_entropy_nd_ann" gives a new value of nb_errors_local
        last_npts_eff += last_npts_eff_local; // each call to "compute_entropy_nd_ann" uses an effective nb of points
    }
    avg /= n_windows;
    var /= n_windows;
    var -= avg*avg;
    
    *S = avg;
    last_std = sqrt(var);
    nb_errors_total += nb_errors;
//    printf("[compute_entropy_ann] : %d effective points\n", last_npts_eff);
    free(x_new); 
//    printf("nb errors = %d\n", nb_errors);
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
*****************************************************************************************/
int compute_relative_entropy_ann(double *x, int nx, double *y, int ny, int mx, int my, int px, int py, int stride, int k, double *H)
{    double *x_new, *y_new;
     double avg=0.0, var=0.0, tmp;
     register int j;
#ifndef TEST_EMBED
    register int i,l,d;
#endif       
     int    n_new, nx_new, ny_new, n_windows=1;
#ifdef NAN
     *H=NAN;
#endif
     if ((mx<1) || (my<1)) return(printf("[compute_relative_entropy_ann] : mx and my must be at least 1 !\n"));
     if ((px<1) || (py<1)) return(printf("[compute_relative_entropy_ann] : px and py must be at least 1 !\n"));
     if (stride<1)         return(printf("[compute_relative_entropy_ann] : stride must be at least 1 !\n"));
     if (k<1)              return(printf("[compute_relative_entropy_ann] : k has to be equal or larger than 1.\n"));
    
     if (mx*px != my*py)   return(printf("[compute_relative_entropy_ann] : resulting pdfs do not have the same dimensions!\n"));
     n_new = mx*px;
     
     // Theiler:
     nx_new    = (nx-nx%stride)/stride - (px-1); // size of a single dataset in x
     ny_new    = (ny-ny%stride)/stride - (py-1); // size of a single dataset in y
     n_windows = stride; // there are n_window different datasets to work with
     
     if (nx_new<2*k)       return(printf("[compute_relative_entropy_ann] : not enough points (nx/stride=%d vs k=%d)\n", nx/stride, k));
     if (ny_new<2*k)       return(printf("[compute_relative_entropy_ann] : not enough points (ny/stride=%d vs k=%d)\n", ny/stride, k));
     
     x_new     = (double*)calloc(n_new*nx_new, sizeof(double)); // embedded x
     y_new     = (double*)calloc(n_new*ny_new, sizeof(double)); // embedded y
     
     nb_errors=0; last_npts_eff=0;  // global variables
     for (j=0; j<n_windows; j++)    // loop over independant windows
     {  
#ifdef TEST_EMBED
        Theiler_embed_old(x+(stride*(px-1)+j), nx, mx, px, stride, stride, x_new, nx_new);
        Theiler_embed_old(y+(stride*(py-1)+j), ny, my, py, stride, stride, y_new, ny_new);        
#else
        for (i=0; i<nx_new; i++)    // loop over points in 1 window
        {  for (l=0; l<mx; l++)     // loop over existing dimensions in x
              for (d=0; d<px; d++)  // loop over embedding  in x
                 x_new[i + (l + d*mx)*nx_new] = x[l*nx + j + n_windows*i + stride*(px-1-d)];
        }
        for (i=0; i<ny_new; i++)    // loop over points in 1 window
        {  for (l=0; l<my; l++)     // loop over existing dimensions in y
              for (d=0; d<py; d++)  // loop over embedding  in y
                 y_new[i + (l + d*my)*ny_new] = y[l*ny + j + n_windows*i + stride*(py-1-d)];
        }
#endif           
        if (USE_PTHREAD>0) // if we want multithreading
            tmp = compute_relative_entropy_2xnd_ann_threads(x_new, nx_new, y_new, ny_new, n_new, k,
                                get_cores_number(GET_CORES_SELECTED));            
        else
            tmp = compute_relative_entropy_2xnd_ann        (x_new, nx_new, y_new, ny_new, n_new, k);     
            
        avg += tmp;
        var += tmp*tmp;
        nb_errors += nb_errors_local;
        last_npts_eff += last_npts_eff_local; // each call to "compute_entropy_nd_ann" uses an effective nb of points
     }
     avg /=n_windows;
     var /=n_windows;
     var -= avg*avg;
     
     *H  = avg;
     last_std = sqrt(var);
    
     free(x_new);
     free(y_new);
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
/************************************************************************************/
int compute_mutual_information_ann(double *x, double *y, int nx, int mx, int my, int px, int py, int stride, int k, double *I1, double *I2)
{	double *x_new;
	double mi1=0.0, mi2=0.0, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;
	register int j;
#ifndef TEST_EMBED
    register int i,l,d;
#endif  	
	int    n, nx_new, pp, n_windows=1;

#ifdef NAN
    *I1=NAN; *I2=NAN;
#endif

#ifdef DEBUG
     debug_trace("[compute_mutual_information_ann] signal x", x, nx, mx, px, stride, k);
     debug_trace("[compute_mutual_information_ann] signal y", y, nx, my, py, stride, k);
#endif
     
	if ((mx<1) || (my<1)) return(printf("[compute_mutual_information_ann] : mx and my must be at least 1 !\n"));
	if ((px<1) || (py<1)) return(printf("[compute_mutual_information_ann] : px and py must be at least 1 !\n"));
	if (stride<1)         return(printf("[compute_mutual_information_ann] : stride must be at least 1 !\n"));
    if (k<1)              return(printf("[compute_mutual_information_ann] : k has to be at least 1.\n"));
    
	n	   = mx*px+my*py;
	pp     = (px>py) ? px : py;	  // largest past

    // Theiler method
	nx_new    = (nx-nx%stride)/stride - (pp-1); // size of a single dataset
	n_windows = stride; // there are n_window different datasets to work with
	
    if ((nx_new)<2*k)
    {   return(printf("[compute_mutual_information_nd_ann] : not enough points (nx/stride=%d vs k=%d)\n", nx/stride, k));
	}
	
	x_new     = (double*)calloc(n*nx_new, sizeof(double)); // we work with a unique dataset

    nb_errors=0; last_npts_eff=0;
	for (j=0; j<n_windows; j++)  // loop over independant windows
	{	
#ifdef TEST_EMBED
        Theiler_embed_old(x+(stride*(pp-1)+j), nx, mx, px, stride, stride, x_new,              nx_new);
        Theiler_embed_old(y+(stride*(pp-1)+j), nx, my, py, stride, stride, x_new+mx*px*nx_new, nx_new);
#else
	    for (i=0; i<nx_new; i++) // loop over points in 1 window
		{	for (l=0; l<mx; l++) // loop over existing dimensions in x
            for (d=0; d<px; d++) // loop over embedding  in x
                x_new[i +         (d + l*px)*nx_new] = x[l*nx + j + n_windows*i + stride*(pp-1-d)];
			for (l=0; l<my; l++) // loop over existing dimensions in y
            for (d=0; d<py; d++) // loop over embedding  in y
                x_new[i + (mx*px + d + l*py)*nx_new] = y[l*nx + j + n_windows*i + stride*(pp-1-d)];
		}
#endif

#ifdef DEBUG
         debug_trace("[compute_mutual_information_ann] signal x_new", x_new, nx_new, n, px, stride, k);
#endif
        
        if (USE_PTHREAD>0) // if we want multithreading
        {//   printf("------- MI multithreading\n");
            nb_errors += compute_mutual_information_2xnd_ann_threads(x_new, nx_new, 
                                mx*px, my*py, k, &mi1, &mi2, get_cores_number(GET_CORES_SELECTED));
        }
        else                // single threaded algorithms:
//        if (MI_algo&COUNTING_TEST)  // 2021-12-10 : tests
        {   nb_errors += compute_mutual_information_direct_ann(x_new, nx_new, mx*px, my*py, k, &mi1, &mi2);
        }
/*        else
        { if (MI_algo&COUNTING_ANN) // should we use the version with ANN counting?
		    {   printf("------- MI with ANN counting\n");
		        nb_errors += compute_mutual_information_2xnd_ann_new(x_new, nx_new, mx*px, my*py, k, &mi1, &mi2);
		    }
          else // use the legacy version (proved to be working, and more efficient for lower dimensionality)
            {   printf("------- MI with legacy counting\n");
                nb_errors += compute_mutual_information_2xnd_ann    (x_new, nx_new, mx*px, my*py, k, &mi1, &mi2);
            }
        }
*/        
		avg1 += mi1;    var1 += mi1*mi1;
        avg2 += mi2;	var2 += mi2*mi2;
		last_npts_eff += last_npts_eff_local;
	}
	avg1 /= n_windows;  var1 /= n_windows;
	avg2 /= n_windows;  var2 /= n_windows;
	var1 -= avg1*avg1;
	var2 -= avg2*avg2;
	*I1  = avg1;        last_std  = sqrt(var1);
	*I2  = avg2;		last_std2 = sqrt(var2);
	
	free(x_new);
	return(nb_errors);
} /* end of function "compute_mutual_information_ann" **********************************/



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
 *************************************************************************************/
int compute_partial_MI_ann(double *x, double *y, double *z, int npts, int *dim, int stride, int k, double *I1, double *I2)
{	register int j, shift, n, pp, mx=dim[0], my=dim[1], mz=dim[2], px=dim[3+0], py=dim[3+1], pz=dim[3+2];
#ifndef TEST_EMBED
    register int i,l,d;
#endif  
	int    npts_new, n_windows;
	double *x_new;
    double mi1, mi2, avg1=0.0, avg2=0.0, var1=0.0, var2=0.0;

#ifdef NAN
    *I1=NAN; *I2=NAN;
#endif
    
	if (stride<1)                   return(printf("[compute_partial_MI_ann] : stride has to be equal or larger than 1.\n"));
	if ((px<1) || (py<1) || (pz<1)) return(printf("[compute_partial_MI_ann] : embedding dimensions must be at least 1 !\n"));
    if ((mx<1) || (my<1) || (mz<1)) return(printf("[compute_partial_MI_ann] : initial dimensions of data must be at least 1 !\n"));
    if (k<1)                        return(printf("[compute_entropy_MI_ann] : k must be at least 1 !\n"));
    
	n	   = mx*px + my*py + mz*pz;  // total dimension of the new variable
	pp     = (px>py) ? px : py;     // who has the largest past ?
	pp     = (pp>pz) ? pp : pz;

    // Theiler
	npts_new  = (npts - npts%stride)/stride - (pp-1); // size of a single dataset
	n_windows = stride; // there are n_window different datasets to work with
    
	x_new  = (double*)calloc(n*npts_new, sizeof(double));
    if (x_new==NULL) return(printf("[compute_partial_MI_ann] : memory allocation problem !\n"
                                    "\tplease check parameters (especially dimensions, total is %d)\n", n));
	
	shift  = (pp-1)*stride;      // present is shifted (embedding will be in the past, ie, causal)
	
	nb_errors=0; last_npts_eff=0;   // global variables
    for (j=0; j<n_windows; j++)     // loop over independant windows
	{
#ifdef TEST_EMBED
        Theiler_embed_old(x+shift+j, npts, mx, px, stride, stride, x_new                         , npts_new);
        Theiler_embed_old(z+shift+j, npts, mz, pz, stride, stride, x_new+(      (mx*px)*npts_new), npts_new);
        Theiler_embed_old(y+shift+j, npts, my, py, stride, stride, x_new+((mx*px+mz*pz)*npts_new), npts_new);
#else
	    for (i=0; i<npts_new; i++)  // loop over points in 1 window
        {	for (d=0; d<px; d++)    // loop over embedding in x
            {   for (l=0; l<mx; l++) // loop over dimensions in x
                x_new[i +                 (d + l*px)*npts_new] = x[j + n_windows*i + shift - stride*d + l*npts];
            }
            for (d=0; d<pz; d++)    // loop over embedding in z
            {   for (l=0; l<mz; l++) // loop over dimensions in z
                x_new[i +         (mx*px + d + l*pz)*npts_new] = z[j + n_windows*i + shift - stride*d + l*npts];
            }
            for (d=0; d<py; d++) 
            {   for (l=0; l<my; l++)
                x_new[i + (mx*px + mz*pz + d + l*py)*npts_new] = y[j + n_windows*i + shift - stride*d + l*npts];
            }
        }
#endif
        if (USE_PTHREAD>0) // if we want multithreading
        {   nb_errors += compute_partial_MI_direct_ann_threads(
                        x_new, npts_new, mx*px, my*py, mz*pz, k, &mi1, &mi2, get_cores_number(GET_CORES_SELECTED));
        }
        else                // single threaded algorithms:
        {   nb_errors += compute_partial_MI_engine_ann(
                        x_new, npts_new, mx*px, my*py, mz*pz, k, &mi1, &mi2);
        }

        avg1 += mi1;    var1 += mi1*mi1;
		avg2 += mi2;	var2 += mi2*mi2;
		last_npts_eff += last_npts_eff_local;
	}
	avg1 /= n_windows;  var1 /= n_windows;
	avg2 /= n_windows;	var2 /= n_windows;
	var1 -= avg1*avg1;
	var2 -= avg2*avg2;

	*I1  = avg1;        last_std  = sqrt(var1);
	*I2  = avg2;    	last_std2 = sqrt(var2);

    if (nb_errors!=0) printf("[compute_partial_MI_nd_ann] error, %d points were problematic.\n", nb_errors);
    free(x_new);
	return(nb_errors);
} /* end of function "compute_partial_MI_ann" */


