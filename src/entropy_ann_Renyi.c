/*
 *  entropy_ANN_Renyi.c
 *
 *  Created by Nicolas Garnier on 05/10/2014.
 *  Copyright 2014-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2014-10-05 : new function "compute_Renyi_nd_ann"
 *  2021-12-15 : using multithreading
 *  2021-12-15 : returning std of estimator (and other statistics)
 *  2022-06-09 : nw with enahanced samplings and increments
 *
 */

#include <math.h>                   // for fabs
#include <gsl/gsl_sf.h>             // for psi digamma function

#include "library_commons.h"        // definitions of nb_errors, and stds
#include "library_matlab.h"         // compilation for Matlab
#include "ANN_wrapper.h"            // for ANN library (in C++)
#include "samplings.h"              // created 2021-12-17 (to factor some operations)
#include "entropy_ann_threads.h"
#include "entropy_ann_threads_Renyi.h"


/****************************************************************************************/
/* computes Renyi entropy, using nearest neighbor statistics (Leonenko 2008)            */
/* this is derived from Leonenko, Pronzato - Ann. Statist. 36 p2153-82 (2008)           */
/*                                                                                      */
/* this version is for n-dimentional systems, and uses ANN library with kd-tree         */
/*                                                                                      */
/* this version does not support embedding per se, but can be used by a wrapper which   */
/* embedds the data (the function "compute_Renyi_ann()" will do that)                   */
/*                                                                                      */
/* x   contains all the data, which is of size n*nx                                     */
/* nx  is the number of points in time                                                  */
/* n   is the dimensionality                                                            */
/* q   is the order of the Renyi entropy to compute                                     */
/* k   is the number of neighbors to consider                                           */
/*                                                                                      */
/* data is ordered like this :                                                          */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)               */
/* 2012-02-27, fork from "compute_entropy_nd_ann"                                       */
/* 2014-10-05, new function "compute_Renyi_nd_ann"                                      */
/* 2017-11-29, improvements (not necessary)                                             */
/* 2021-12-15, added test for robustness                                                */
/****************************************************************************************/
double compute_Renyi_nd_ann(double *x, int nx, int n, double q, int k)
{   register int i;
//    int    l;
    double epsilon, *epsilon_z;
    double h=0.00;
    //	FILE *fe;
    
    epsilon_z = (double*)calloc(n, sizeof(double));
    init_ANN(nx, n, k, SINGLE_TH);
    create_kd_tree(x, nx, n);
    nb_errors_local=0;
    
    for (i=0; i<nx; i++)
    {   epsilon = ANN_find_distance_in(i, n, k, 0);  // 2021-12-01 pthread
        
        if (epsilon==0) nb_errors_local++;
        else /* estimateur de l'integrale (XII.153) : esperance de la grandeur suivante : */
            h = h + exp((double)n*(1.0-q)*log(epsilon));
        //            fprintf(fe,"%f\n",epsilon);
        
    }
    
    if (nb_errors_local<nx)
    {   h = h/(double)(nx-nb_errors_local); /* normalisation de l'esperance */
    
        /* ci-après, application de la formule (XII.153) : */
        h = log(h);
        h = h + log(gsl_sf_gamma((double)k)) - log(gsl_sf_gamma((double)(k+1.0-q)));
        h = h/((double)1.0-q);
        h = h + log((double)(nx-1-nb_errors_local)) // replaced nx by nx-1-nb_errors (2017-11-29)
                    + n*log((double)2.0);           // XII.153
    }
    else    // big trouble
    {   h = my_NAN;   
    }
    
    
    /* free pointers de taille n=dimension de l'espace : */
    free(epsilon_z);
    free_ANN(SINGLE_TH);
    
    last_npts_eff_local = nx-nb_errors_local;
    return(h);
} /* end of function "compute_Renyi_nd_ann" *********************************************/





/****************************************************************************************/
/* computes Renyi entropy, using nearest neighbor statistics (Leonenko 2008)	        */
/*                                                                                      */
/* this version is for m-dimentional systems, with eventually some stride/embedding	    */
/*																			            */
/* x      contains all the data, which is of size nx in time						    */
/* nx     is the number of points in time											    */
/* m	  indicates the (initial) dimensionality of x								    */
/* p	  indicates how many points to take in time (in the past) (embedding)           */
/* stride is the time lag between 2 consecutive points to be considered in time		    */
/* q      is the order of the Renyi entropy                                             */
/* k      nb of neighbors to be considered										        */
/* method to choose embedding (0), increments (1), or higher order increments (2)       */
/*																			            */
/* data is ordered like this :													        */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)				*/
/*																			            */
/* this function is a  wrapper to the functions :									    */
/*	- compute_Renyi_nd_nns													            */
/*																			            */
/* 2014-10-05 : first version, copy from compute_entropy_ann                            */
/* 2021-12-15 : using multi-threading                                                   */
/* 2021-12-15 : returning std, nb_errors and nb_eff                                     */
/* 2022-06-08 : new samplings, untested                                                 */
/* 2022-06-09 : now also for various increments (via "method" parameter)                */
/****************************************************************************************/
int compute_Renyi_ann(double *x, int npts, int m, int p, int tau, double q, int tau_Theiler, int N_eff, int N_realizations, int k, int method, double *S)
{
    register int j, p_new;
    double *x_new, S_tmp, avg=0.0, var=0.0;
    int    N_real_max=0;
	samp_param  sp = { .Theiler=tau_Theiler, .N_eff=N_eff, .N_real=N_realizations};
	gsl_permutation *perm_real, *perm_pts;
	
    *S = my_NAN;

    if ((m<1) || (p<1))     return(printf("[compute_Renyi_ann] : m and p must be at least 1 !\n"));
    if (tau<1)              return(printf("[compute_Renyi_ann] : stride must be at least 1 !\n"));
    if ((k<1))              return(printf("[compute_Renyi_ann] : k must be at least 1 !\n"));
    if (q==1)               return(printf("[compute_Renyi_ann] : you asked for q=1, which makes the code diverge...\n"));
    if ((method<0) || (method>2))
                            return(printf("[compute_Renyi_ann] : method must be 0, 1 or 2 !\n"));
	
	// adapt embedding dimension according to method:
	if (method==0)  p_new = p;  // time-embedding (not increments)
	else {          p_new = 1;  // increments (regular or averaged)
	                p    += 1;  // convention: increments of order 1 (p) require 2 (p+1) points in order to be computed
	     }
        
    // additional checks and auto-adjustments of parameters:
    N_real_max = set_sampling_parameters(npts, p, tau, &sp, "compute_entropy_ann_N");
    if (N_real_max<1)       return(printf("[compute_Renyi_ann] : aborting !\n"));
    if (sp.N_eff < 2*k)     return(printf("[compute_Renyi_ann] : N_eff=%d is too small compared to k=%d)\n", sp.N_eff, k));
    
    x_new  = (double*)calloc(m*p_new*sp.N_eff, sizeof(double));
    
    perm_real = create_unity_perm(N_real_max);  if (sp.type>=3) shuffle_perm(perm_real);
    perm_pts = create_unity_perm(sp.N_eff_max);     // for random sampling
    
    nb_errors=0; last_npts_eff=0;
    for (j=0; j<sp.N_real; j++)     // loop over independant windows
    {   if (sp.type>=3) shuffle_perm(perm_pts);
    
        if (method==0)      // regular embedding (not increments)
            Theiler_embed(x+(tau*(p-1)+perm_real->data[j]), npts, m, p, tau, sp.Theiler, perm_pts->data, x_new, sp.N_eff);
        else if (method==1) // regular increments
            increments   (x+(tau*(p-1)+perm_real->data[j]), npts, m, p, tau, sp.Theiler, perm_pts->data, x_new, sp.N_eff);
        else if (method==2) // averaged increments
            incr_avg     (x+(tau*(p-1)+perm_real->data[j]), npts, m, p, tau, sp.Theiler, perm_pts->data, x_new, sp.N_eff);
            
        if (USE_PTHREAD==1) 
            S_tmp = compute_Renyi_nd_ann_threads(x_new, sp.N_eff, m*p_new, q, k, get_cores_number(GET_CORES_SELECTED));
        else
            S_tmp = compute_Renyi_nd_ann        (x_new, sp.N_eff, m*p_new, q, k);
            
        avg  += S_tmp;
        var  += S_tmp*S_tmp;
        nb_errors += nb_errors_local; // each call to "compute_entropy_nd_ann" gives a new value of nb_errors_local
        last_npts_eff += last_npts_eff_local;      
    }
    avg /= sp.N_real;
    var /= sp.N_real;   var -= avg*avg;
    
    *S = avg;
    last_std = sqrt(var);
    
    nb_errors_total += nb_errors;
    last_samp=sp;
    
    free(x_new);
    free_perm(perm_real);    free_perm(perm_pts);
    return(nb_errors);
} /* end of function "compute_Renyi_ann" *************************************/


