/*
 *  entropy_ann_entropy_threads.c
 *
 *  Created by Nicolas Garnier on 2021/12/02.
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

// thread arguments and outputs types:
struct thread_args
    {   int core;    // keep track of the current thread number/id
        int i_start; // begining of subset of points to work on
        int i_end;   // end      of subset of points
        int n;       // dimensionality of data
        int k;       // nb of neighbors to search for
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
void *threaded_Shannon_entropy_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(sizeof(struct thread_output),1); // allocate heap memory for this thread's results
    register int i;
    double eps=0.0, l_hs=0.0;
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        n        = args->n,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;

    for (i=i_start; i<i_end; i++)
    {   eps = ANN_find_distance_in(i, n, k, core);
        if (is_zero(eps)) l_errors++;
        else l_hs += log(eps);
    }
    
    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->h_sum    = l_hs;
        
//    printf("\t\t%d-%d=%d\n", i_start, i_end, i_end-i_start);
//        free(args);
    pthread_exit(out);
}




/****************************************************************************************/
/* computes Shannon entropy, using nearest neighbor statistics (2004)                   */
/* this is derived from PRE 69 066138 (2004)                                            */
/*                                                                                      */
/* this version is for n-dimentional systems, and uses ANN library with kd-tree         */
/*                                                                                      */
/* this version does not support embedding per se, but can be used by a wrapper which   */
/* embedds the data (the function "compute_entropy_ann()" being exactly this)           */
/*                                                                                      */
/* x        contains all the data, which is of size npts*nx                             */
/* npts     is the number of points in time                                             */
/* n        is the dimensionality of the data                                           */
/* k        is the number of neighbors to consider                                      */
/* nb_cores is the nb of threads to use                                                 */
/*          if ==-1, then it will be set to max nb of threads (auto-detect)             */
/*                                                                                      */
/* data is ordered like this :                                                          */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)               */
/*                                                                                      */
/* 2012-02-27  fork from "compute_entropy_ann"                                          */
/* 2021-11-26  first multi-threads version                                              */
/****************************************************************************************/
double compute_entropy_nd_ann_threads(double *x, int npts, int n, int k, int nb_cores)
{	register int core, npts_eff_min;
	double h=0.00;
	int n_total=0; // just for sanity check
	int ret;
    
    if (nb_cores<1) set_cores_number(nb_cores); // auto-detect, if asked for
    // 2022-12-13: all other threads number manipulation should be done outside of this engine function!
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
        my_arguments[core].k = k;
        ret=pthread_create(&thread[core], NULL, threaded_Shannon_entropy_func, (void *)&my_arguments[core]);
        if (ret!=0)
        {   printf("[compute_entropy_nd_ann_threads] TROUBLE! couldn't create thread!\n");
            return(my_NAN); 
        }
    }
    for (core=0; core<nb_cores; core++)
    {   pthread_join(thread[core], (void**)&my_outputs[core]);
        h += my_outputs[core]->h_sum;
        nb_errors_local += my_outputs[core]->n_errors;
        n_total += my_outputs[core]->n_eff; // just for sanity
        free(my_outputs[core]);
    }
    
    // sanity check:
    if (n_total!=npts) printf("[compute_entropy_nd_ann_threads] TROUBLE! npts altered!\n");
    
    if (nb_errors_local>=npts) h=my_NAN;   // big trouble
    else // we can get an estimate
    {   h = h/(double)(npts-nb_errors_local); /* normalisation de l'esperance */
	
        /* normalisation : */
        h = h*(double)n;
        h = h + gsl_sf_psi_int(npts-nb_errors_local) - gsl_sf_psi_int(k);
        h = h + (double)n*log((double)2.0);	/* notre epsilon est le rayon, et pas le diametre de la boule */
    }
//    printf("nb errors local : %d, DBL_MIN = %g\n", nb_errors_local, DBL_MIN*1e13);
    
	/* free pointers de taille n=dimension de l'espace : */
	free_ANN(nb_cores);
	last_npts_eff_local = npts-nb_errors_local;
    return(h);
} /* end of function "compute_entropy_nd_ann" *************************************/
