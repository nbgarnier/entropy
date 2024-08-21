/*
 *  entropy_ANN_Renyi.c
 *
 *  Created by Nicolas Garnier on 05/10/2014.
 *  Copyright 2014-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2014-10-05 : new function "compute_Renyi_nd_ann"
 *  2021-12-15 : new function using pthreads
 *
 */

#include <math.h>                   // for log
#include <gsl/gsl_sf.h>             // for psi digamma function
#include <pthread.h>

#include "entropy_ann_threads.h"    // for mamangement of threads number
#include "ANN_wrapper.h"            // for ANN library (in C++)
#include "library_commons.h"        // contains the definition of nb_errors, and stds

// thread arguments and ouputs types:
struct thread_args
    {   int core;       // keep track of the current thread number/id
        int i_start;    // begining of subset of points to work on
        int i_end;      // end      of subset of points
        int n;          // dimensionality of data
        int q;          // order of Renyi entropy
        int k;          // nb of neighbors to search for
    };

struct thread_output
    {   int n_eff;
        int n_errors;
        double h_sum;
    };



/****************************************************************************************/
/* function to be used by "compute_Renyi_nd_ann_threads"                                */
/*                                                                                      */
/* 2021-12-15, fork from compute_Renyi_nd_ann() function, first multi-threads version   */
/****************************************************************************************/
void *threaded_Renyi_entropy_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; 
    struct thread_output *out = calloc(sizeof(struct thread_output),1); 
    register int i;
    double eps=0.0, l_hs=0.0;
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        n        = args->n,
        q        = args->q,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
        
    for (i=i_start; i<i_end; i++)
    {   eps = ANN_find_distance_in(i, n, k, core);  // 2021-12-01 pthread
        
        if (eps==0) l_errors++;
        else /* estimateur de l'integrale (XII.153) : esperance de la grandeur suivante : */
            l_hs += exp((double)n*(1.0-q)*log(eps));        
    }

    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->h_sum    = l_hs;

    pthread_exit(out);
}


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
/* 2021-12-15, fork from compute_Renyi_nd_ann() function, first multi-threads version   */
/****************************************************************************************/
double compute_Renyi_nd_ann_threads(double *x, int npts, int n, double q, int k, int nb_cores)
{   register int core, npts_eff_min;
    double h=0.00;
    int    ret=0, n_total=0;

    if (nb_cores<1) set_cores_number(nb_cores); // auto-detect
    nb_cores=get_cores_number(GET_CORES_SELECTED);
    
	init_ANN(npts, n, k, get_cores_number(GET_CORES_SELECTED)); 	
    create_kd_tree(x, npts, n);
    nb_errors_local=0;
    
    // partage des points entre threads:
    npts_eff_min   = (npts - (npts%nb_cores))/nb_cores;  // nb pts mini dans chaque thread
    
    // define threads:
    pthread_t    thread[nb_cores];
    struct thread_args   my_arguments[nb_cores];
    struct thread_output *my_outputs[nb_cores];

    for (core=0; core<nb_cores; core++)
    {   my_arguments[core].core    = core;
        my_arguments[core].i_start = core*npts_eff_min;
        my_arguments[core].i_end   = (core+1)*npts_eff_min;
        if (core==(nb_cores-1)) my_arguments[core].i_end = npts;  // last thread: will work longer!
        my_arguments[core].n = n;
        my_arguments[core].q = q;
        my_arguments[core].k = k;
        ret=pthread_create(&thread[core], NULL, threaded_Renyi_entropy_func, (void *)&my_arguments[core]);
        if (ret!=0) printf("[compute_Renyi_nd_ann_threads] TROUBLE! cannot create thread!\n");
    }
    for (core=0; core<nb_cores; core++)
    {   pthread_join(thread[core], (void**)&my_outputs[core]);
        h += my_outputs[core]->h_sum;
        nb_errors_local += my_outputs[core]->n_errors;
        n_total += my_outputs[core]->n_eff; // just for sanity
        free(my_outputs[core]);
    }

    // sanity check:
    if (n_total!=npts) printf("[compute_Renyi_nd_ann_threads] TROUBLE! npts altered!\n");
    
    if (nb_errors_local>=npts) h=my_NAN;   // big trouble
    else // we can get an estimate
    {   h = h/(double)(npts-nb_errors_local); /* normalisation de l'esperance */
    
        /* ci-après, application de la formule (XII.153) : */
        h = log(h);
        h = h + log(gsl_sf_gamma((double)k)) - log(gsl_sf_gamma((double)(k+1.0-q)));
        h = h/((double)1.0-q);
        h = h + log((double)(npts-1-nb_errors_local)) // replaced nx by nx-1-nb_errors (2017-11-29)
                + n*log((double)2.0);                 // XII.153
    }
    
    free_ANN(nb_cores);
    last_npts_eff_local = npts-nb_errors_local;
    return(h);
} /* end of function "compute_Renyi_nd_ann_threads" *************************************/


