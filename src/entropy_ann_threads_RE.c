/*
 *  entropy_ann_threads_RE.c
 *
 *  Created by Nicolas Garnier on 2021/12/09.
 *  Copyright 2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *
 */
 
#include <stdlib.h>
#include <stdio.h>      // for printf
#include <string.h>     // for malloc
#include <math.h>       // for log 
#include <gsl/gsl_sf.h> // for psi digamma function
#include <pthread.h>

#include "entropy_ann_threads.h"
#include "ANN_wrapper.h"
#include "library_commons.h"   // for is_zero()
#include "library_matlab.h"

// thread arguments and ouputs types:
struct thread_args
    {   int core;    // keep track of the current thread number/id
        int i_start; // begining of subset of points to work on
        int i_end;   // end      of subset of points
        int nx;      // nb of data points in x
        int n;       // dimensionality of data
        int k;       // nb of neighbors to search for
        double *x;   // pointer to the data
    };

struct thread_output
    {   int n_eff;
        int n_errors;
        double h_sum;
    };



/****************************************************************************************/
/* function to be used by "compute_entropy_nd_ann_threads"                              */
/*                                                                                      */
/* 2021-11-26  first multi-threads version                                              */
/****************************************************************************************/
void *threaded_relative_entropy_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(sizeof(struct thread_output),1); // allocate heap memory for this thread's results
    register int i, j;
    double eps=0.0, l_hs=0.0;
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        nx       = args->nx,
        n        = args->n,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
    double *x, *x_central;

    x         = args->x;               // pointer to the data (read only, so thread-safe)
    x_central = (double *)calloc(n, sizeof(double));
    
    for (i=i_start; i<i_end; i++)
    {   for (j=0; j<n; j++) x_central[j] = x[i + j*nx];
    
        eps = ANN_find_distance_ex(x_central, n, k, core); 
        if (is_zero(eps)) l_errors++;
        else l_hs += log(eps);
    }
    
    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->h_sum    = l_hs;

    free(x_central);
    pthread_exit(out);
}




/**************************************************************************************
 * computes relative Shannon entropy H(x||y), using nearest neighbor statistics
 * this is derived from: Leonenko et al, Annals of Statistics 36 (5) pp2153–2182 (2008)
 * page 2163
 *
 * this version is for n-dimentional systems, and uses ANN library with kd-tree
 *
 * this version does not support embedding per se, but can be used by a wrapper which
 * embedds the data (the function "compute_relative_entropy_ann()" being exactly this)
 *
 * x   contains the data, distributed from f(x). x is of size n*nx
 * nx  is the number of observations of x (number of points in time)
 * y   contains the data, distributed from g(y). y is of size n*ny
 * ny  is the number of observations of y
 * n   is the dimensionality (of both x and y)
 * k   is the number of neighbors to consider
 *
 * data is ordered like this :
 * x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)
 *
 * 2017-11-29, fork from "compute_entropy_nd_ann"
 * 2021-12-09, fork from "compute_relative_entropy_2xnd_ann"
 ***************************************************************************************/
double compute_relative_entropy_2xnd_ann_threads(double *x, int nx, double *y, int ny, int n, int k, int nb_cores)
{   register int core, npts_eff_min, n_total=0;
    double h=0.00;

    // eventually correct the number of threads requested:
    if (nb_cores<1) set_cores_number(nb_cores); // auto-detect
    nb_cores = get_cores_number(GET_CORES_SELECTED);
    // partition points between threads:
    npts_eff_min   = (nx - (nx%nb_cores))/nb_cores;  // nb pts mini dans chaque thread
    // define threads:
    pthread_t    thread[nb_cores];
    struct thread_args   my_arguments[nb_cores];
    struct thread_output *my_outputs[nb_cores];
  
    init_ANN(ny, n, k, get_cores_number(GET_CORES_SELECTED));
    create_kd_tree(y, ny, n);
    nb_errors_local=0; 
    
    for (core=0; core<nb_cores; core++)
    {   my_arguments[core].core    = core;
        my_arguments[core].i_start = core*npts_eff_min;
        my_arguments[core].i_end   = (core+1)*npts_eff_min;
        if (core==(nb_cores-1)) my_arguments[core].i_end = nx;  // last thread: will work longer!
        my_arguments[core].nx = nx;
        my_arguments[core].n = n;
        my_arguments[core].k = k;
        my_arguments[core].x = x;
        pthread_create(&thread[core], NULL, threaded_relative_entropy_func, (void *)&my_arguments[core]);
    }
    for (core=0; core<nb_cores; core++)
    {   pthread_join(thread[core], (void**)&my_outputs[core]);
        h += my_outputs[core]->h_sum;
        nb_errors_local += my_outputs[core]->n_errors;
        n_total += my_outputs[core]->n_eff; // just for sanity
        free(my_outputs[core]);
    }

    // sanity check:
    if (n_total!=nx) printf("[compute_relative_entropy_2xnd_ann_threads] TROUBLE! npts altered!\n");

    if (nb_errors_local>=nx) h = my_NAN;   // big trouble
    else
    {   h /= (double)(nx-nb_errors_local); /* normalisation de l'esperance */
     
        /* ci-après, normalisation : */
        h *= (double)n;
        h += gsl_sf_psi_int(ny-nb_errors_local) - gsl_sf_psi_int(k); 
        h += (double)n*log((double)2.0);     /* notre epsilon est le rayon, et pas le diametre de la boule */
    }
    
    /* free pointers : */
    free_ANN(nb_cores);
    last_npts_eff_local = nx-nb_errors_local; 
    return(h);
} /* end of function "compute_relative_entropy_2xnd_ann_threads" ******************************/


