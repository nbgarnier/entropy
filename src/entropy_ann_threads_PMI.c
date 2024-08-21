/*
 *  entropy_ann_threads_MI.c
 *
 *  this file contains threaded "engine" functions that perform counting based on
 *  the ANN library : http://www.cs.umd.edu/~mount/ANN/
 *
 *  this file is included in "entropy_ann.c"
 *
 *  Created by Nicolas B. Garnier on 2021/12/09.
 *  Copyright 2012-2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *  these functions are wrapped within the Cython code,
 *  where they can be called by setting a flag (|=0x0100) in the algo choice
 *
 *  2021-12-09 : forked from "entropy_ann_new.c" and "entropy_ann_threads_MI.c"
 *  2021-12-14 : now using either NBG or ANN counting
 */

#include <stdlib.h>
#include <stdio.h>      // for printf
#include <string.h>     // for malloc
#include <math.h>       // for log 
#include <gsl/gsl_sf.h> // for psi digamma function
#include <pthread.h>

#include "entropy_ann.h"
#include "entropy_ann_threads.h"
#include "ANN_wrapper.h"
#include "nns_count.h"          // counting functions (2019-01-23)
#include "math_tools.h"
#include "library_commons.h"   // for is_zero()
#include "library_matlab.h"

#define noTIMING
#define DEBUG
#define noDEBUG_EXPORT

#ifdef TIMING
#include <time.h>
#endif

// thread arguments and ouputs types:
struct thread_args
    {   int core;       // keep track of the current thread number/id
        int i_start;    // begining of subset of points to work on
        int i_end;      // end      of subset of points
        int npts;       // length (in points) of the data to work on
        int mx;         // dimensionality of first process
        int my;         // dimensionality of second process
        int mz;         // dimensionality of third process
        int k;          // nb of neighbors to search for
        double *x;      // pointer to the data
        double *y;      // for NBG counting only: pointer to the data
        int *indices_y; // for NBG counting only
        int *ind_inv_y; // for NBG counting only
        int *unit_perm; // for NBG counting only
    };

struct thread_output
    {   int n_eff;
        int n_errors;
        double I_sum;
    };

  
/****************************************************************************************/
/* function to be used by "compute_partial_MI_direct_ann_thread"                        */
/* algorithm 1 from Kraskov et al                                                       */
/* using NBG counting                                                                   */
/*                                                                                      */
/* 2021-12-14  first multi-threads version                                              */
/****************************************************************************************/
void *threaded_PMI_algo1_NBG_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(sizeof(struct thread_output),1); // allocate heap memory for this thread's results
    register int i, d;
    double *yi;
//    double *epsilon_z;
    double epsilon=0.0, l_Is=0.0;       // pointers to tmp variables
    double *x, *y;                      // pointer to the data (from parameters)
     int    *indices_y, *ind_inv_y, *unit_perm;
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        npts     = args->npts,
        mx       = args->mx,
        my       = args->my,
        mz       = args->mz,
        n        = mx+my+mz,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
    int n_xz, n_z, n_zy;

#ifdef TIMING
    clock_t t1, t2;
    double time_alloc, time_0=0, time_1=0, time_2=0;
#endif

    yi        = calloc(n, sizeof(double));
//    epsilon_z = calloc(n, sizeof(double));
    x         = args->x;               // pointer to the data (read only, so thread-safe)
    y         = args->y;               // pointer to the data (read only, so thread-safe)
    indices_y = args->indices_y;       
    ind_inv_y = args->ind_inv_y;       
    unit_perm = args->unit_perm;       

#ifdef TIMING
    t2=clock();
    time_alloc = (double)(t2-t1)/CLOCKS_PER_SEC;
#endif 
 
#ifdef DEBUG_EXPORT
    FILE *fic;
    char filename[64];
    sprintf(filename, "MI_thread_algo1_core%d_th%d.dat", core, get_cores_number(GET_CORES_SELECTED));
    fic = fopen(filename, "wt");
#endif     
    
// old PMI:
    for (i=i_start; i<i_end; i++)
    {	for (d=0; d<n; d++) yi[d] = x[d*npts + i];	/* composantes zi de z */
		
        epsilon = ANN_find_distance_in(i, n, k, 2*core); // 2019-01-22. // 2021-12-01 pthread
        
/*        epsilon    = my_max(epsilon_vec,    mx+my+mz);
        epsilon_xz = my_max(epsilon_vec,    mx+mz); // 2019-01-30
        epsilon_z  = my_max(epsilon_vec+mx, mz);
        epsilon_zy = my_max(epsilon_vec+mx, my+mz);
*/

		if (epsilon>0)
		{   n_xz = count_nearest_neighbors_nd_algo1(x, npts, mx+mz, i,            unit_perm,    yi, epsilon);
            n_z  = count_nearest_neighbors_nd_algo1(y, npts, mz,    ind_inv_y[i], indices_y, yi+mx, epsilon);
			n_zy = count_nearest_neighbors_nd_algo1(y, npts, mz+my, ind_inv_y[i], indices_y, yi+mx, epsilon);

			l_Is += gsl_sf_psi_int(n_z+1) - gsl_sf_psi_int(n_xz+1) - gsl_sf_psi_int(n_zy+1);
		}
        else l_errors++; // (epsilon<=0)
	} // algo_1, end of loop on i

    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->I_sum    = l_Is;

//    free(epsilon_z); 
    free(yi);   
    pthread_exit(out);
} /* end of function "threaded_PMI_algo1_NBG_func" */



/****************************************************************************************/
/* function to be used by "compute_partial_MI_direct_ann_thread"                        */
/* algorithm 1 from Kraskov et al                                                       */
/* using ANN counting                                                                   */
/*                                                                                      */
/* 2021-12-09  first multi-threads version                                              */
/****************************************************************************************/
void *threaded_PMI_algo1_ANN_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(sizeof(struct thread_output),1); // allocate heap memory for this thread's results
    register int i, d;
    double *yi;
    double *epsilon_z;
    double epsilon=0.0, l_Is=0.0;       // pointers to tmp variables
    double *x;                          // pointer to the data (from parameters)
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        npts     = args->npts,
        mx       = args->mx,
        my       = args->my,
        mz       = args->mz,
        n        = mx+my+mz,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
    int n_xz, n_z, n_zy;

#ifdef TIMING
    clock_t t1, t2;
    double time_alloc, time_0=0, time_1=0, time_2=0;
#endif
    
    yi        = calloc(n, sizeof(double));
    epsilon_z = calloc(n, sizeof(double));
    x         = args->x;               // pointer to the data (read only, so thread-safe)

#ifdef TIMING
    t2=clock();
    time_alloc = (double)(t2-t1)/CLOCKS_PER_SEC;
#endif 
 
#ifdef DEBUG_EXPORT
    FILE *fic;
    char filename[64];
    sprintf(filename, "MI_thread_algo1_core%d_th%d.dat", core, get_cores_number(GET_CORES_SELECTED));
    fic = fopen(filename, "wt");
#endif     
    
// old PMI:
    for (i=i_start; i<i_end; i++)
    {	for (d=0; d<n; d++) yi[d] = x[d*npts + i];	/* composantes zi de z */
		
        ANN_marginal_distances_ex(yi, n, k, epsilon_z, 2*core);  
        epsilon=my_max(epsilon_z, n);

/*        epsilon    = my_max(epsilon_vec,    mx+my+mz);
        epsilon_xz = my_max(epsilon_vec,    mx+mz); // 2019-01-30
        epsilon_z  = my_max(epsilon_vec+mx, mz);
        epsilon_zy = my_max(epsilon_vec+mx, my+mz);
*/

		if (epsilon>0)
		{   n_xz = ANN_count_nearest_neighbors_nd_tree1(yi,    epsilon, 2*core); // 2022-02-07: added factor 2 in core counting
            n_z  = ANN_count_nearest_neighbors_nd_tree2(yi+mx, epsilon, 2*core);
            n_zy = ANN_count_nearest_neighbors_nd_tree3(yi+mx, epsilon, 2*core);

            if (my_max(epsilon_z+mx, mz)==epsilon)      // k-th nn is at the boundary along (z)
                n_z -=1;
            if (my_max(epsilon_z, mx+mz)==epsilon)      // k-th nn is at the boundary along (xz)
                n_xz -=1;
            if (my_max(epsilon_z+mx, mz+my)==epsilon)   // k-th nn is at the boundary along (zy)
                n_zy -=1;

			l_Is += gsl_sf_psi_int(n_z+1) - gsl_sf_psi_int(n_xz+1) - gsl_sf_psi_int(n_zy+1);
		}
        else l_errors++; // (epsilon<=0)
	} // algo_1, end of loop on i

    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->I_sum    = l_Is;

    free(epsilon_z); free(yi);   
    pthread_exit(out);
} /* end of function "threaded_PMI_algo1_ANN_func" */



/****************************************************************************************/
/* function to be used by "compute_partial_MI_direct_ann_thread"                        */
/* algorithm 2 from Kraskov et al                                                       */
/* using NBG counting                                                                   */
/* (algo 2 should be restricted to threads with even indexes)                           */
/*                                                                                      */
/* 2021-12-14  first multi-threads version                                              */
/****************************************************************************************/
void *threaded_PMI_algo2_NBG_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(sizeof(struct thread_output),1); // allocate heap memory for this thread's results
    register int i, d;
    double *yi, *epsilon_vec;
//    double epsilon=0.0;
    double epsilon_xz, epsilon_z, epsilon_zy, l_Is=0.0;
    double *x, *y;
    int    *indices_y, *ind_inv_y, *unit_perm;
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        npts     = args->npts,
        mx       = args->mx,
        my       = args->my,
        mz       = args->mz,
        n        = mx+my+mz,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
    int n_xz, n_z, n_zy;
    
    yi        = calloc(n, sizeof(double));
    epsilon_vec = calloc(n, sizeof(double));
    x         = args->x;               // pointer to the data (read only, so thread-safe)
    y         = args->y;
    indices_y = args->indices_y;       
    ind_inv_y = args->ind_inv_y;       
    unit_perm = args->unit_perm;   

    for (i=i_start; i<i_end; i++)
    {    for (d=0; d<n; d++) yi[d] = x[d*npts + i];    /* composantes zi de z */
        
        ANN_marginal_distances_ex(yi, n, k, epsilon_vec, 2*core+1); // new 2019. // 2021-12-01 pthread
//        epsilon    = my_max(epsilon_vec,    mx+my+mz);
        epsilon_xz = my_max(epsilon_vec,    mx+mz); // 2019-01-30
        epsilon_z  = my_max(epsilon_vec+mx, mz);
        epsilon_zy = my_max(epsilon_vec+mx, my+mz);

        if ((epsilon_xz*epsilon_z*epsilon_zy)>0) // more serious then (epsilon>0)
        {   n_xz = count_nearest_neighbors_nd_algo2(x, npts, mx+mz, i,            unit_perm, yi,    epsilon_xz);
            n_z  = count_nearest_neighbors_nd_algo2(y, npts, mz,    ind_inv_y[i], indices_y, yi+mx, epsilon_z);
            n_zy = count_nearest_neighbors_nd_algo2(y, npts, mz+my, ind_inv_y[i], indices_y, yi+mx, epsilon_zy);

            l_Is += gsl_sf_psi_int(n_z) - gsl_sf_psi_int(n_xz) - gsl_sf_psi_int(n_zy);
        }
        else l_errors++; // (epsilon<=0)
    } 

    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->I_sum    = l_Is;

    free(epsilon_vec); free(yi);
    pthread_exit(out);
} /* end of "threaded_PMI_algo2_NBG_func" ****************************************************/



/****************************************************************************************/
/* function to be used by "compute_partial_MI_direct_ann_thread"                        */
/* using ANN counting                                                                   */
/* algorithm 2 from Kraskov et al                                                       */
/* (algo 2 should be restricted to threads with even indexes)                           */
/*                                                                                      */
/* 2021-12-09  first multi-threads version                                              */
/****************************************************************************************/
void *threaded_PMI_algo2_ANN_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(sizeof(struct thread_output),1); // allocate heap memory for this thread's results
    register int i, d;
    double *yi, *epsilon_vec;
//    double epsilon=0.0;
    double epsilon_xz, epsilon_z, epsilon_zy, l_Is=0.0;
    double *x;
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        npts     = args->npts,
        mx       = args->mx,
        my       = args->my,
        mz       = args->mz,
        n        = mx+my+mz,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
    int n_xz, n_z, n_zy;
    
    yi        = calloc(n, sizeof(double));
    epsilon_vec = calloc(n, sizeof(double));
    x         = args->x;               // pointer to the data (read only, so thread-safe)

    for (i=i_start; i<i_end; i++)
    {    for (d=0; d<n; d++) yi[d] = x[d*npts + i];    /* composantes zi de z */
        
        ANN_marginal_distances_ex(yi, n, k, epsilon_vec, 2*core+1); // new 2019. // 2021-12-01 pthread
//        epsilon    = my_max(epsilon_vec,    mx+my+mz);
        epsilon_xz = my_max(epsilon_vec,    mx+mz); // 2019-01-30
        epsilon_z  = my_max(epsilon_vec+mx, mz);
        epsilon_zy = my_max(epsilon_vec+mx, my+mz);

        if ((epsilon_xz*epsilon_z*epsilon_zy)>0) // more serious then (epsilon>0)
        {   n_xz = ANN_count_nearest_neighbors_nd_tree1(yi,    epsilon_xz, 2*core+1);
            n_z  = ANN_count_nearest_neighbors_nd_tree2(yi+mx, epsilon_z,  2*core+1);
            n_zy = ANN_count_nearest_neighbors_nd_tree3(yi+mx, epsilon_zy, 2*core+1);

            l_Is += gsl_sf_psi_int(n_z) - gsl_sf_psi_int(n_xz) - gsl_sf_psi_int(n_zy);
        }
        else l_errors++; // (epsilon<=0)
    } 

    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->I_sum    = l_Is;

    free(epsilon_vec); free(yi);
    pthread_exit(out);
} /* end of "threaded_PMI_algo2_ANN_func" ****************************************************/



/***************************************************************************************
 * computes partial mutual information, as defined by Frenzel and Pompe
 * using nearest neighbor statistics (Grassberger 2004)
 * this version is for (m+p+q)-dimentional systems	 
 *
 * (ordering of data is a bit strange, please use the wrapper instead)							
 *
 * for the definition, and the estimator, see the article from Frenzel, Pompe 
 *		 PRL 99, 204101 (2007)
 *		 "Partial Mutual Information for Coupling Analysis of Multivariate Time Series"
 *
 * I(X,Y|Z) = part of the MI I(X,Y) which is not in Z
 *
 * x      contains all the data, which is of size (m+p+q)*npts, that is, huge
 * npts     is the number of points in time	
 * m      is the dimension of x (can be the nb of point in the past)
 * p      is the dimension of y
 * q      is the dimension of z, the conditioning variable
 * k      is the number of neighbors to consider
 *
 * data in x is ordered like this :										
 * x1(t=1)...x1(t=npts-1) x2(t=0) ... x2(t=npts-2) ... xn(t=0) ... xn(t=npts-2)
 * components of X are first, and then are components of conditioning variable Z,
 * and then are components of second variable Y                                         
 * If considering time embedding, dimensions are from future to past
 *
 * this function is inspired from "compute_transfer_entropy_direct_nns()
 * this function uses subfunctions :								
 *	- search_nearest_neighbors							
 *	- count_nearest_neighbors_nd_algo1						
 *	- count_nearest_neighbors_nd_algo2n	(2n and not 2 !!!)	
 *
 * Some sanity tests are executed only ifdef DEBUG (compile with -dDEBUG to have the tests)
 *
 * Note that algorithm 2 is not working, and is coded to give a result equal to 0 
 *											
 * 2012-01-03 : first version (nns, no kd-tree)
 * 2012-02-29 : kd-tree version (with ANN library)
 * 2012-03-12 : algo 2 commented out (not working)
 * 2012-03-12 : bug correction in allocation of *ind_toto
 * 2012-09-04 : bug correction in counting neighbors : bug was appearing when m>1
 * 2019-01-31 : "new" version using counting with new functions using ANN library
 * 2020-02-27 : using global variable "nb_errors_local" to return nb of errors
 * 2021-12-09 : first multithreaded version (ANN counting)
 * 2021-12-14 : now with NBG counting
 ***************************************************************************************/
int compute_partial_MI_direct_ann_threads(double *x, int npts, int mx, int my, int mz, 
                    int k, double *I1, double *I2, int nb_cores)
{
	register int d, i, j, core=0;
	int    npts_eff_min, nb_points_check1=0, nb_points_check2=0;
	int    n, nb_errors_1=0, nb_errors_2=0;
	int	   *indices_x, *ind_inv_x, *indices_y, *ind_inv_y, *unit_perm=NULL;
	double h1=0.0, h2=0.0;
	double *y, *toto;   // will be allocated
	double *y_tmp;      // pure pointer
    
    *I1=my_NAN;     *I2=my_NAN;

	n = mx+my+mz; /* total dimensionality of data */
	if (n<3) return(printf("[compute_partial_MI_direct_ann_threads] : not enough dimensions in data (at least 3 required)\n"));
	
    /* we allocate memory */
	indices_x = (int*)calloc(npts, sizeof(int));	/* permutation des indices pour passer de x à x sorted */
	ind_inv_x = (int*)calloc(npts, sizeof(int));  /* permutation inverse */
    indices_y = (int*)calloc(npts, sizeof(int));  /* only to count neighbors    */
    ind_inv_y = (int*)calloc(npts, sizeof(int));  /* only to count neighbors    */
    y         = (double*)calloc(npts*(mz+my), sizeof(double));
    toto      = (double*)calloc(npts, sizeof(double));
    
    if ( (toto==NULL) || (ind_inv_y==NULL) ) return(printf("[compute_partial_MI_direct_ann_new] : alloc error\n"));
	
    for (i=0; i<npts; i++) indices_x[i]=i;
	QuickSort_double(x, indices_x, 0, npts-1);		/* we sort the data, pointers x and indices_x are modified ! */
	for (i=0; i<npts; i++) ind_inv_x[indices_x[i]]=i;	
	    /* so we have x_sorted[ind_inv_x[i]] = x_unsorted[i] */
	    /* and        x_unsorted[ind[i]] = x_sorted[i]		*/
	for (d=1; d<n; d++) /* we arrange all other dimensions accordingly : */
	{	y_tmp = x+(d*npts);
		for (i=0; i<npts; i++) toto[i]  = y_tmp[indices_x[i]];
		for (i=0; i<npts; i++) y_tmp[i] = toto[i];
	}

	/* to save some time when counting, we order the first dimension of (z,y) now :	*/
	memcpy(y, x+mx*npts, npts*(mz+my)*sizeof(double));
	for (j=0; j<npts; j++) indices_y[j] = j;
	QuickSort_double(y, indices_y, 0, npts-1);   // 2012/01/03 : we only sort first direction
                                            //  of both (z,y) and (z) (without x)
	for (j=0; j<npts; j++) ind_inv_y[indices_y[j]] = j;
	
	if (MI_algo&COUNTING_ANN)
    {	for (d=1; d<(mz+my); d++) /* we arrange all other dimensions accordingly : */
        {   y_tmp = y+d*npts;
            for (i=0; i<npts; i++) toto[i]  = y_tmp[indices_y[i]];
            for (i=0; i<npts; i++) y_tmp[i] = toto[i];
        }
    }
    else // for NBG algo only
    {   unit_perm = (int*)calloc(npts, sizeof(int));
        for (j=0; j<npts; j++) unit_perm[j] = j;
    }

    // pthread house-keeping:
    if (nb_cores<1) set_cores_number(nb_cores);      // we have to auto-detect nb_cores
    nb_cores = get_cores_number(GET_CORES_SELECTED); // and then keep track of this number
    
    if (MI_algo&COUNTING_ANN)                   // for PMI, there will be indeed 3*nb_cores
    {   init_ANN_PMI(npts, mx, my, mz, k, nb_cores);
        create_kd_tree  (x, npts, mx+mz+my);    // full tree
        create_kd_tree_1(x, npts, mx+mz);       // XZ tree
        create_kd_tree_2(y, npts, mz);          // Z tree
        create_kd_tree_3(y, npts, mz+my);       // ZY tree
    }
    else
    {   init_ANN(npts, n, k, nb_cores*3);       // 2021-12-14: check if 3*nb_cores is necessary or not
        create_kd_tree(x, npts, n);
	}
  
// partage des points entre threads:
    npts_eff_min   = (npts - (npts%nb_cores))/nb_cores; // nb pts mini dans chaque thread
    
    // define threads:
    pthread_t    thread_algo1[nb_cores];                // there are nb_cores threads for algo 1
    pthread_t    thread_algo2[nb_cores];                // and also nb_cores other threads for algo 2
    struct thread_args    my_arguments[nb_cores];       // parameters are the same for both algos
    struct thread_output *my_outputs_algo1[nb_cores];   // but outputs are differents
    struct thread_output *my_outputs_algo2[nb_cores];
    
    for (core=0; core<nb_cores; core++)
    {   my_arguments[core].core    = core;
        my_arguments[core].i_start = core*npts_eff_min;
        my_arguments[core].i_end   = (core+1)*npts_eff_min;
        if (core==(nb_cores-1)) my_arguments[core].i_end = npts;  // last thread: will work longer!
        my_arguments[core].npts = npts; // thread independent
        my_arguments[core].mx = mx;     // thread independent
        my_arguments[core].my = my;     // thread independent
        my_arguments[core].mz = mz;     // thread independent
        my_arguments[core].k = k;       // thread independent
        my_arguments[core].x = x;       // thread independent
        my_arguments[core].y = y;       // thread independent
        my_arguments[core].indices_y = indices_y;       // thread independent
        my_arguments[core].ind_inv_y = ind_inv_y;       // thread independent
        my_arguments[core].unit_perm = unit_perm;       // thread independent
        
        if (MI_algo & MI_ALGO_1)
        {   if (MI_algo&COUNTING_ANN) 
            pthread_create(&thread_algo1[core], NULL, threaded_PMI_algo1_ANN_func, (void *)&my_arguments[core]);
            else
            pthread_create(&thread_algo1[core], NULL, threaded_PMI_algo1_NBG_func, (void *)&my_arguments[core]);
        }
        else h1=my_NAN;
        if (MI_algo & MI_ALGO_2)
        {   if (MI_algo&COUNTING_ANN)
            pthread_create(&thread_algo2[core], NULL, threaded_PMI_algo2_ANN_func, (void *)&my_arguments[core]);
            else
            pthread_create(&thread_algo2[core], NULL, threaded_PMI_algo2_NBG_func, (void *)&my_arguments[core]);
        }
        else h2=my_NAN;
    }

    for (core=0; core<nb_cores; core++)
    {   if (MI_algo & MI_ALGO_1) 
        {   pthread_join(thread_algo1[core], (void**)&my_outputs_algo1[core]);
            h1               += my_outputs_algo1[core]->I_sum;
            nb_errors_1      += my_outputs_algo1[core]->n_errors;
            nb_points_check1 += my_outputs_algo1[core]->n_eff;
            free(my_outputs_algo1[core]);
        }
        if (MI_algo & MI_ALGO_2) 
        {   pthread_join(thread_algo2[core], (void**)&my_outputs_algo2[core]);
            h2               += my_outputs_algo2[core]->I_sum;
            nb_errors_2      += my_outputs_algo2[core]->n_errors;
            nb_points_check2 += my_outputs_algo2[core]->n_eff;
            free(my_outputs_algo2[core]);
        }
    }

    if ( (MI_algo & MI_ALGO_1) && (nb_points_check1 != npts) )
        printf("[compute_partial_MI_direct_ann_threads] nb pts not conserved!\n");
    if ( (MI_algo & MI_ALGO_2) && (nb_points_check2 != npts) )
        printf("[compute_partial_MI_direct_ann_threads] nb pts not conserved!\n");

    nb_errors_local = nb_errors_1 + nb_errors_2; // 2020-02-27: errors may be counted twice if both algos are used
    

#ifdef DEBUG	
    if (nb_errors_1>0) printf("[compute_partial_MI_direct_ann_threads] %d errors encountered in algo 1: be carefull, the result may not be valid !\n", nb_errors_1);
    if (nb_errors_2>0) printf("[compute_partial_MI_direct_ann_threads] %d errors encountered in algo 2: be carefull, the result may not be valid !\n", nb_errors_2);
#endif
	if (npts<=nb_errors_local) // test re-organized on 2019-01-21
    {   printf("[compute_partial_MI_direct_ann_threads] pb : npts=%d points in dataset, %d errors. aborting...\n", 
                npts, nb_errors);
    }
    else
    {   h1 = h1/(double)(npts-nb_errors_1); /* normalisation de l'esperance */
        h2 = h2/(double)(npts-nb_errors_2); /* normalisation de l'esperance */
	
        /* ci-après, normalisation finale : */
        h1 = h1 + gsl_sf_psi_int(k);
        h2 = h2 + gsl_sf_psi_int(k); // - (double)2.0/(double)k;
                                     // 2 = 3-1, with 3 marginal directions in the combinaison.
                // 2019-02-01 : removed a factor, which should not have been there!!! XII.259
        *I1 = h1;
        *I2 = h2;
    }
    
	/* then we restore initial pointers (as they were before being ordered by QuickSort) */
	for (d=0; d<n; d++)
	{	y_tmp = x+(d*npts);
		for (i=0; i<npts; i++) y_tmp[i] = y[ind_inv_x[i]];
		for (i=0; i<npts; i++) y[i]     = y_tmp[i];	
	}	

	free(indices_x); free(ind_inv_x);
    free(indices_y); free(ind_inv_y);
	free(y); free(toto);
    if (MI_algo&COUNTING_ANN) free_ANN_PMI(nb_cores);
    else {                    free_ANN(3*nb_cores); // 2022-02-07: added factor 3
                              free(unit_perm);
         }
	return(nb_errors_local);	
} /* end of function "compute_partial_MI_direct_ann_threads" */

