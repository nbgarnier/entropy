/*
 *  entropy_ann_threads_MI.c
 *
 *  this file contains threaded "engine" functions that perform counting based on
 *  the ANN library : http://www.cs.umd.edu/~mount/ANN/
 *
 *  this file is included in "entropy_ann.c"
 *
 *  Created by Nicolas B. Garnier on 2021/12/02.
 *  Copyright 2012-2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *  these functions are wrapped within the Cython code,
 *  where they can be called by setting a flag (|=0x0100) in the algo choice
 *
 *  2021-12-02 : forked from entropy_ann_new.c
 *  2021-12-09 : working fine
 *  2021-12-13 : unified function, for either NBG or ANN counting
 */

#include <stdlib.h>
#include <stdio.h>              // for printf
#include <string.h>             // for malloc
#include <gsl/gsl_sf.h>         // for psi digamma function
#include <pthread.h>

#include "entropy_ann.h"
#include "entropy_ann_threads.h"
#include "ANN_wrapper.h"
#include "nns_count.h"          // counting functions (2019-01-23)
#include "math_tools.h"
#include "library_commons.h"    // for is_zero()
#include "verbosity.h"          // for advanced warnings/error mmanagement
#include "library_matlab.h"

#define noTIMING
#define noDEBUG
#define noDEBUG_EXPORT

#ifdef TIMING
    #include "timings.h"
#else 
    #define tic()
    #define toc(x) // 0.0
#endif

// extern pthread_mutex_t mutex_annkFRSearch;  // 2021-12-07, NBG: experimentation
// extern pthread_mutex_t mutex_strong;        // 2021-12-07, NBG: experimentation

// thread arguments and ouputs types:
struct thread_args
    {   int core;       // keep track of the current thread number/id
        int i_start;    // begining of subset of points to work on
        int i_end;      // end      of subset of points
        int npts;       // length (in points) of the data to work on
        int mx;         // dimensionality of first process
        int my;         // dimensionality of second process
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
/* function to be used by "compute_mutual_information_2xnd_ann_threads"                 */
/* algorithm 1 from Kraskov et al                                                       */
/* using NBG counting                                                                   */
/*                                                                                      */
/* 2021-12-13  first multi-threads version                                              */
/****************************************************************************************/
void *threaded_MI_algo1_NBG_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(1, sizeof(struct thread_output)); // allocate heap memory for this thread's results
    register int i, d;
    double *yi;
    double epsilon=0.0, l_Is=0.0;       // pointers to tmp variables
    double *x, *y;                      // pointer to the data (from parameters)
    int    *indices_y, *ind_inv_y, *unit_perm;
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        npts     = args->npts,
        mx       = args->mx,
        my       = args->my,
        n        = mx+my,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
    int n_x1, n_y1;  /* for algo 1 */
#ifdef TIMING
    double time_alloc=0, time_0=0, time_1=0, time_2=0;
#endif
    
    tic();   
    yi = calloc(n, sizeof(double));
//    epsilon_z = calloc(n, sizeof(double));
    x         = args->x;               // pointer to the data (read only, so thread-safe)
    y         = args->y;               // pointer to the data (read only, so thread-safe)
    indices_y = args->indices_y;       
    ind_inv_y = args->ind_inv_y;       
    unit_perm = args->unit_perm;       
    toc(&time_alloc); 
 
#ifdef DEBUG_EXPORT
    FILE *fic;
    char filename[64];
    sprintf(filename, "MI_thread_algo1_NBG_core%d_th%d.dat", core, get_cores_number(GET_CORES_SELECTED));
    fic = fopen(filename, "wt");
#endif     
    
    for (i=i_start; i<i_end; i++)
    {   for (d=0; d<n; d++) yi[d] = x[d*npts + i];	/* composantes zi de z */

#ifdef DEBUG_EXPORT
        fprintf(fic, "%3d, center: ", i);
        for (d=0; d<n; d++) fprintf(fic, "%2f ", yi[d]); // center point
#endif 

        // 2021-12-01: pthread (2*core for algo 1 and 2*core+1 for algo 2)
        tic(); // t1=clock();
//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation        
        epsilon = ANN_find_distance_in(i, n, k, 2*core); 
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation        
        // 2019-12-18: maybe search_ANN_external is faster? to check!
        toc(&time_0);
        
#ifdef DEBUG_EXPORT
        fprintf(fic, " \teps: ");
        fprintf(fic, "%2f ", epsilon);  // epsilon
        fclose(fic); fopen(filename, "at");
#endif          
        if (epsilon>0)
        { //  pthread_mutex_lock(&mutex_strong); // 2021-12-07, NBG: experimentation

            tic();
// ANN:     n_x1 = ANN_count_nearest_neighbors_nd_tree1(yi,    epsilon, 2*core);
            n_x1 = count_nearest_neighbors_nd_algo1(x, npts, mx,           i,  unit_perm, yi,   epsilon);
// NBG: we need to prepare and share properly x,y, unit_perm, indices_y and ind_inv_y
			toc(&time_1);
            
            tic();
// ANN:     n_y1 = ANN_count_nearest_neighbors_nd_tree2(yi+mx, epsilon, 2*core); 
        	n_y1 = count_nearest_neighbors_nd_algo1(y, npts, my, ind_inv_y[i], indices_y, yi+mx, epsilon);
            toc(&time_2);
       	
#ifdef DEBUG
            if (n_x1*n_y1<=0)
            { printf("\t[threaded_MI_algo1_NBG_func], core %d\t i=%d, %2d and %2d (new)\n", core, i, n_x1, n_y1);
              printf("\t        epsilon = %f\t x (center) : ", epsilon);
              for (int zozo=0; zozo<n; zozo++) printf("%f ", yi[zozo]); 
              printf("\n");
            }
            else
#endif
            l_Is += - gsl_sf_psi_int(n_x1+1) - gsl_sf_psi_int(n_y1+1);
            
#ifdef DEBUG_EXPORT
            fprintf(fic, "\tnx_x,y (1): ");
            fprintf(fic, "%2d %2d ", n_x1, n_y1); // counting
            fclose(fic); fopen(filename, "at");
#endif      
		}
        else 
        {   l_errors++; // (epsilon==0)
            if (lib_verbosity>2)
            {   printf("\talgo_1, NBG count., core %d\t i=%d, \tepsilon: %f\n", core, i, epsilon);
            }
        }
#ifdef DEBUG_EXPORT
        fprintf(fic, "\n");
#endif          

	} // algo_1, end of loop on i

#ifdef DEBUG_EXPORT
        fclose(fic);
#endif  
    
    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->I_sum    = l_Is;

    // 2021-12-07: added line below to clean up memory:
    free(yi);
#ifdef TIMING
    printf("[threaded_MI_algo1_NBG_func] timings: %f for alloc., %f for k-search, %f for n_x, %f for n_y\n", time_alloc, time_0, time_1, time_2);
#endif    
    pthread_exit(out);
} /* end of function "threaded_MI_algo1_NBG_func" */



/****************************************************************************************/
/* function to be used by "compute_mutual_information_2xnd_ann_threads"                 */
/* algorithm 1 from Kraskov et al                                                       */
/* using ANN counting                                                                   */
/*                                                                                      */
/* 2021-12-02  first multi-threads version                                              */
/****************************************************************************************/
void *threaded_MI_algo1_ANN_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(1, sizeof(struct thread_output)); // allocate heap memory for this thread's results
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
        n        = mx+my,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
    int n_x1=0, n_y1=0;  /* for algo 1 */
#ifdef TIMING
    double time_alloc=0, time_0=0, time_1=0, time_2=0;
#endif
    
    tic();   
    yi        = calloc(n, sizeof(double));
    epsilon_z = calloc(n, sizeof(double));
    x         = args->x;               // pointer to the data (read only, so thread-safe)
    toc(&time_alloc); 
 
#ifdef DEBUG_EXPORT
    FILE *fic;
    char filename[64];
    sprintf(filename, "MI_thread_algo1_ANN_core%d_th%d.dat", core, get_cores_number(GET_CORES_SELECTED));
    fic = fopen(filename, "wt");
#endif     
    
    for (i=i_start; i<i_end; i++)
    {
        for (d=0; d<n; d++) yi[d] = x[d*npts + i];	/* composantes zi de z */

#ifdef DEBUG_EXPORT
        fprintf(fic, "%3d, center: ", i);
        for (d=0; d<n; d++) fprintf(fic, "%2f ", yi[d]); // center point
#endif 

        // 2021-12-01: pthread (2*core for algo 1 and 2*core+1 for algo 2)
        tic(); // t1=clock();
//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation        
        ANN_marginal_distances_ex(yi, n, k, epsilon_z, 2*core); 
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation        
        // 2019-12-18: maybe search_ANN_external is faster? to check!
        toc(&time_0);

        epsilon=my_max(epsilon_z, n); // 2019-12-17: tested identical to ANN_find_distance_in(i, n, k)
        
#ifdef DEBUG_EXPORT
        fprintf(fic, " \teps: ");
        fprintf(fic, "%2f ", epsilon);  // epsilon
        fclose(fic); fopen(filename, "at");
#endif          
        if (epsilon>0)
        { //  pthread_mutex_lock(&mutex_strong); // 2021-12-07, NBG: experimentation

            tic();
            n_x1 = ANN_count_nearest_neighbors_nd_tree1(yi,    epsilon, 2*core); 
            toc(&time_1);
            
            tic();
        	n_y1 = ANN_count_nearest_neighbors_nd_tree2(yi+mx, epsilon, 2*core); 
            toc(&time_2);

        //	pthread_mutex_unlock(&mutex_strong); // 2021-12-07, NBG: experimentation
        	// 2019-12-17, below is a correction because we want < and not <= for algo 1:
        	if (my_max(epsilon_z, mx)==epsilon) // the k-th neighbor is at the boundary in the x dimension
        		 n_x1 -=1; // so it was counted wrongly
        	else n_y1 -=1; // it was in the y dimensions
        	// 2019-12-17: both case are mutualy exclusive: tested OK
        	// 2019-12-17: note this correction is imperfect, because it uses only the k-th neighbor
        	// 				and maybe there are other points at the boundary... (?)
       	
#ifdef DEBUG
            if (n_x1*n_y1<=0)
            { printf("\t[threaded_MI_algo1_ANN_func], core %d\t i=%d, %2d and %2d (new)\n", core, i, n_x1, n_y1);
              printf("\t        epsilon = %f\t x (center) : ", epsilon);
              for (int zozo=0; zozo<n; zozo++) printf("%f ", yi[zozo]); 
              printf("\n");
            }
            else
#endif
            l_Is += - gsl_sf_psi_int(n_x1+1) - gsl_sf_psi_int(n_y1+1);
            
#ifdef DEBUG_EXPORT
            fprintf(fic, "\tnx_x,y (1): ");
            fprintf(fic, "%2d %2d ", n_x1, n_y1); // counting
            fclose(fic); fopen(filename, "at");
#endif      
		}
        else 
        {   l_errors++; // (epsilon==0)
            if (lib_verbosity>2)
            {   printf("\talgo_1, ANN count., core %d\t i=%d, %2d and %2d\tepsilon: %f\n", core, i, n_x1, n_y1, epsilon);
            }
        }
#ifdef DEBUG_EXPORT
        fprintf(fic, "\n");
#endif          

	} // algo_1, end of loop on i

#ifdef DEBUG_EXPORT
        fclose(fic);
#endif  
    
    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->I_sum    = l_Is;

    // 2021-12-07: added line below to clean up memory:
    free(epsilon_z); free(yi);
#ifdef TIMING
    printf("[threaded_MI_algo1_ANN_func] timings: %f for alloc., %f for k-search, %f for n_x, %f for n_y\n", time_alloc, time_0, time_1, time_2);
#endif    
    pthread_exit(out);
} /* end of function "threaded_MI_algo1_ANN_func" */



/****************************************************************************************/
/* function to be used by "compute_mutual_information_2xnd_ann_threads"                 */
/* algorithm 2 from Kraskov et al                                                       */
/* using NBG counting                                                                   */
/* (algo 2 should be restricted to threads with even indexes)                           */
/*                                                                                      */
/* 2021-12-13  first multi-threads version                                              */
/****************************************************************************************/
void *threaded_MI_algo2_NBG_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(1, sizeof(struct thread_output)); // allocate heap memory for this thread's results
    register int i, d;
    double *yi, *epsilon_z;
    double epsilon_1, epsilon_2, l_Is=0.0;
#ifdef DEBUG
    double epsilon=0.0;
#endif    
    double *x, *y;
    int    *indices_y, *ind_inv_y, *unit_perm;
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        npts     = args->npts,
        mx       = args->mx,
        my       = args->my,
        n        = mx+my,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
    int n_x2=0, n_y2=0;  /* for algo 1 */
    
    yi        = calloc(n, sizeof(double));
    epsilon_z = calloc(n, sizeof(double));
    x         = args->x;               // pointer to the data (read only, so thread-safe)
    y         = args->y;
    indices_y = args->indices_y;       
    ind_inv_y = args->ind_inv_y;       
    unit_perm = args->unit_perm;   
    
#ifdef DEBUG_EXPORT
    FILE *fic;
    char filename[64];
    sprintf(filename, "MI_thread_algo2_NBG_core%d_th%d.dat", core, get_cores_number(GET_CORES_SELECTED));
    fic = fopen(filename, "wt");
#endif  
    
    for (i=i_start; i<i_end; i++)
    {   for (d=0; d<n; d++) yi[d] = x[d*npts + i];	/* composantes zi de z */        

#ifdef DEBUG_EXPORT
        fprintf(fic, "%3d, center: ", i);
        for (d=0; d<n; d++) fprintf(fic, "%2f ", yi[d]); // center point
#endif   
  
        ANN_marginal_distances_ex(yi, n, k, epsilon_z, 2*core+1); // new 2019-01-28 // 2021-12-01 pthread    
        epsilon_1 = my_max(epsilon_z,    mx); // 2019-01-30
        epsilon_2 = my_max(epsilon_z+mx, my);
        
#ifdef DEBUG_EXPORT
        fprintf(fic, " \teps_z: ");
        for (d=0; d<n; d++) fprintf(fic, "%2f ", epsilon_z[d]); // epsilon
        fclose(fic); fopen(filename, "at");
#endif     

        if (epsilon_1*epsilon_2>0) // let's be strict
        {   n_x2 = count_nearest_neighbors_nd_algo2(x, npts, mx,           i,  unit_perm, yi,    epsilon_1);
            n_y2 = count_nearest_neighbors_nd_algo2(y, npts, my, ind_inv_y[i], indices_y, yi+mx, epsilon_2);

#ifdef DEBUG
            if (n_x2*n_y2<=0)
            { printf("\t[threaded_MI_algo2_NBG_func], core %d\t i=%d, %2d and %2d (new)\n", core, i, n_x2, n_y2);
              printf("\t        epsilon = %f\t x (center) : ", epsilon);
              for (int zozo=0; zozo<n; zozo++) printf("%f ", yi[zozo]); 
              printf("\n");
            }
            else
#endif
            l_Is += - gsl_sf_psi_int(n_x2) - gsl_sf_psi_int(n_y2);            
            
#ifdef DEBUG_EXPORT
            fprintf(fic, "\tnx_x,y (2): ");
            fprintf(fic, "%2d %2d ", n_x2, n_y2); // counting
            fclose(fic); fopen(filename, "at");
#endif      
            
		}
        else 
        {   l_errors++; // (epsilon==0)
            if (lib_verbosity>2)
            {   printf("\talgo_2, NBG count., core %d\t i=%d, %2d and %2d\tepsilon: %f  %f\n", 
                            core, i, n_x2, n_y2, epsilon_1, epsilon_2);
            }
        }
        
#ifdef DEBUG_EXPORT
        fprintf(fic, "\n");
#endif 
        
	} // algo_2, end of loop on i
    

#ifdef DEBUG_EXPORT
    fclose(fic);
#endif   
    
    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->I_sum    = l_Is;

    free(epsilon_z); free(yi); // added 2022-03-19 (forgotten!)
    pthread_exit(out);
} /* end of "threaded_MI_algo2_NBG_func" ************************************************/



/****************************************************************************************/
/* function to be used by "compute_mutual_information_2xnd_ann_threads"                 */
/* algorithm 2 from Kraskov et al                                                       */
/* using ANN counting                                                                   */
/* (algo 2 should be restricted to threads with even indexes)                           */
/*                                                                                      */
/* 2021-12-02  first multi-threads version                                              */
/****************************************************************************************/
void *threaded_MI_algo2_ANN_func(void *ptr)
{   struct thread_args  *args = (struct thread_args *)ptr; // cast arguments to the usable struct
    struct thread_output *out = calloc(1, sizeof(struct thread_output)); // allocate heap memory for this thread's results
    register int i, d;
    double *yi, *epsilon_z;
//    double epsilon=0.0;
    double epsilon_1, epsilon_2, l_Is=0.0;
    double *x;
    int core     = args->core,
        i_start  = args->i_start,
        i_end    = args->i_end,
        npts     = args->npts,
        mx       = args->mx,
        my       = args->my,
        n        = mx+my,
        k        = args->k,
        n_eff    = i_end-i_start, // how many points in this thread
        l_errors = 0;
    int n_x2=0, n_y2=0;  /* for algo 1 */
    
    yi        = calloc(n, sizeof(double));
    epsilon_z = calloc(n, sizeof(double));
    x         = args->x;               // pointer to the data (read only, so thread-safe)
   
#ifdef DEBUG_EXPORT
    FILE *fic;
    char filename[64];
    sprintf(filename, "MI_thread_algo2_ANN_core%d_th%d.dat", core, get_cores_number(GET_CORES_SELECTED));
    fic = fopen(filename, "wt");
#endif  
    
    for (i=i_start; i<i_end; i++)
    {
//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation               
        for (d=0; d<n; d++) yi[d] = x[d*npts + i];	/* composantes zi de z */
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation        

#ifdef DEBUG_EXPORT
        fprintf(fic, "%3d, center: ", i);
        for (d=0; d<n; d++) fprintf(fic, "%2f ", yi[d]); // center point
#endif   
//pthread_mutex_lock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation        
        ANN_marginal_distances_ex(yi, n, k, epsilon_z, 2*core+1); // new 2019-01-28 // 2021-12-01 pthread
//pthread_mutex_unlock(&mutex_annkFRSearch); // 2021-12-07, NBG: experimentation        
        epsilon_1 = my_max(epsilon_z,    mx); // 2019-01-30
        epsilon_2 = my_max(epsilon_z+mx, my);
        // 2019-12-18: maybe search_ANN_external is faster? to check!
//        epsilon=my_max(epsilon_z, n); // 2019-12-17: tested identical to ANN_find_distance_in(i, n, k)
// note 2021-12-03: epsilon is unused in this function??? to check with regular MI algo!!!

#ifdef DEBUG_EXPORT
        fprintf(fic, " \teps_z: ");
        for (d=0; d<n; d++) fprintf(fic, "%2f ", epsilon_z[d]); // epsilon
        fclose(fic); fopen(filename, "at");
#endif     

        if (epsilon_1*epsilon_2>0) // let's be strict
        {   //pthread_mutex_lock(&mutex_strong); // 2021-12-07, NBG: experimentation
            n_x2 = ANN_count_nearest_neighbors_nd_tree1(yi,    epsilon_1, 2*core+1);
            n_y2 = ANN_count_nearest_neighbors_nd_tree2(yi+mx, epsilon_2, 2*core+1);
            //pthread_mutex_unlock(&mutex_strong); // 2021-12-07, NBG: experimentation
//            printf("\t %2d and %2d (new)\n", n_x2, n_y2);
#ifdef DEBUG
            if (n_x2*n_y2<=0)
            { printf("\t[threaded_MI_algo2_ANN_func], core %d\t i=%d, %2d and %2d (new)\n", core, i, n_x2, n_y2);
//              printf("\t        epsilon = %f\t x (center) : ", epsilon);
              for (int zozo=0; zozo<n; zozo++) printf("%f ", yi[zozo]); 
              printf("\n");
            }
            else
#endif
            l_Is += - gsl_sf_psi_int(n_x2) - gsl_sf_psi_int(n_y2);            
            
#ifdef DEBUG_EXPORT
            fprintf(fic, "\tnx_x,y (2): ");
            fprintf(fic, "%2d %2d ", n_x2, n_y2); // counting
            fclose(fic); fopen(filename, "at");
#endif      
            
		}
        else 
        {   l_errors++; // (epsilon==0)
            if (lib_verbosity>2)
            {   printf("\talgo_2, ANN count., core %d\t i=%d, %2d and %2d\tepsilon: %f  %f\n", 
                        core, i, n_x2, n_y2, epsilon_1, epsilon_2);
            }
        }
        
#ifdef DEBUG_EXPORT
        fprintf(fic, "\n");
#endif 
        
	} // algo_2, end of loop on i
    

#ifdef DEBUG_EXPORT
    fclose(fic);
#endif   
    
    out->n_eff    = n_eff;
    out->n_errors = l_errors;
    out->I_sum    = l_Is;

    free(epsilon_z); free(yi); // added 2022-03-19 (forgotten!)
    pthread_exit(out);
} /* end of "threaded_MI_algo2_ANN_func" ************************************************/



/****************************************************************************************/
/* computes mutual information, using nearest neighbor statistics (Grassberger 2004)    */
/* this is an application of PRE 69 066138 (2004)									    */
/*                                                                                      */
/* this version is for (m+p)-dimentional systems							    		*/
/* and computes information redundency of 2 variables of dimension m and p		    	*/
/*																			            */
/* x       contains all the data, which is of size (m+p)*npts, (possibly huge)			*/
/* npts    is the number of points in time											    */
/* mx, my  (int) are the dimensionalities of the first and second variables	            */
/* k       (int) is the nb of nearest neighbors to consider								*/
/*                                                                                      */
/* I1, I2  (double) are returned values, according to algorithm 1 and 2                 */
/*                                                                                      */
/* nb_cores (int) is the number of threads to use                                       */
/*                                                                                      */
/* data is ordered like this (in case of m and p from some embedding):                  */
/* x1(t=0)...x1(t=npts-1) x2(t=0) ... x2(t=npts-1) ... xn(t=0) ... xn(t=npts-1)			*/
/* components of first variables are first, and then are components of second variable	*/
/*																			            */
/* m and p can also be either:                                                          */
/* - simply the dimensionality of x and y (no embedding)                                */
/* - the products of initial dimensionalities by embedding dimensions                   */
/* For embedding, please use a wrapper                                                  */
/*																			            */
/* this function uses subfunctions :												    */
/*	- search_nearest_neighbors													        */
/*	- count_nearest_neighbors_nd_algo1											        */
/*	- count_nearest_neighbors_nd_algo2n	(2n and not 2 !!!)							    */
/*																			            */
/* 2010-10-31 : first draft version 												    */
/* 2010-11-01 : huge cleaning and new function search_nearest_neighbors				    */
/* 2011-06-22 : functions now returns an integer = nb of errors						    */
/* 2011-11-14 : errors and warnings only tested for and printed if in DEBUG mode		*/
/* 2012-02-05 : algo 2 was working with nns, but is not working with ANN... to explore  */
/*              => algo 2 is commented out (to gain some speed)                         */
/* 2012-03-14 : test : added factor (n-1) in front of psi(npts) for correct             */
/*               normalisation according to Kraskov formula (23)       THEN REMOVED !   */
/*																			            */
/* to-do : test sur erreur grave lorsque nb_error > nb_voisin						    */
/* 2019-01-28 : algo 2 is back!                                                         */
/* 2019-01-30 : using a set of new functions (using ANN) to count the neighbors         */
/*              results looks OK-ish, but there is a severe decrease of performance     */
/*              compared to N.G. trick with pre-sorting                                 */
/* 2019-01-31 : => pre-sorting incorporated on new functions => factor 2 in speed!      */
/* 2019-12-17 : correction for algo 1													*/
/* 2020-02-27 : using global variable "nb_errors_local" to return the nb of errors		*/
/* 2021-12-01 : multithread version development starts, forked from _new (ANN counting) */
/* 2021-12-09 : multithread version working (with ANN counting)                         */
/* 2021-12-13 : multithread version  with NBG counting now available, and working       */
/****************************************************************************************/
int compute_mutual_information_2xnd_ann_threads(double *x, int npts, int mx, int my, 
                    int k, double *I1, double *I2, int nb_cores)
{	char fname[64]="compute_mutual_information_2xnd_ann_threads";
    char message[128];
	register int d, i, j, core;
	int	   *indices_x, *ind_inv_x, *indices_y, *ind_inv_y;
	int    *unit_perm=NULL; // for NBG counting
	int    n, nb_errors_1=0, nb_errors_2=0;
    int     npts_eff_min;
	double h1=0.0, h2=0.0;
	double *y, *toto;
	double *y_tmp;
	int     nb_points_check1=0, nb_points_check2=0;

    *I1 = my_NAN;   *I2 = my_NAN;
    
    n = mx+my; /* total dimensionality of data */
	if (n<2) return(print_error(fname, "not enough dimensions in data (at least 2 required)"));
	
    toto      = (double*)calloc(npts, sizeof(double));  /* pure tmp variable  */
    
	/* x corresponds to the favored dimension d=0 : */
	indices_x = (int*)calloc(npts, sizeof(int));	/* permutation des indices pour passer de x à x sorted */
	ind_inv_x = (int*)calloc(npts, sizeof(int));  /* permutation inverse */
	for (i=0; i<npts; i++) indices_x[i]=i;	
	QuickSort_double(x, indices_x, 0, npts-1);		/* we sort the data, pointers x and indices_x are modified ! */
	for (i=0; i<npts; i++) ind_inv_x[indices_x[i]]=i;
	/* so we have x_sorted[ind_inv_x[i]] = x_unsorted[i] */
	/* and        x_unsorted[ind[i]] = x_sorted[i]		*/
	
	/* we arrange all other dimensions accordingly : */
	for (d=1; d<n; d++)
	{	y_tmp = x+(d*npts);
		for (i=0; i<npts; i++) toto[i]  = y_tmp[indices_x[i]];
		for (i=0; i<npts; i++) y_tmp[i] = toto[i];
	}
	
	/* we allocate memory */
    indices_y = (int*)calloc(npts, sizeof(int));  /* only to count neighbors    */
    ind_inv_y = (int*)calloc(npts, sizeof(int));  /* only to count neighbors    */
	y         = (double*)calloc(npts*my, sizeof(double));
    
	/* to save some time when counting, we order first dimension of second variable y now :	*/
	memcpy(y, x+mx*npts, npts*my*sizeof(double));
	for (j=0; j<npts; j++) indices_y[j] = j;
	QuickSort_double(y, indices_y, 0, npts-1);
		// 2011/07/14 : we only sort direction m which is the first direction of the second vector
	for (j=0; j<npts; j++) ind_inv_y[indices_y[j]] = j;
	if (MI_algo&COUNTING_ANN) 
    {   for (d=1; d<my; d++) /* we arrange all other dimensions accordingly : */
        {   y_tmp = y+d*npts;
            for (i=0; i<npts; i++) toto[i]  = y_tmp[indices_y[i]];
            for (i=0; i<npts; i++) y_tmp[i] = toto[i];
        }
    }
    else
    {   unit_perm = (int*)calloc(npts, sizeof(int));      // unitary permutation (identity)
	    for (j=0; j<npts; j++) unit_perm[j] = j;
    }
    
    // pthread house-keeping:
    if (nb_cores<1) set_cores_number(nb_cores);      // we have to auto-detect nb_cores
    nb_cores = get_cores_number(GET_CORES_SELECTED); // and then keep track of this number
    
    if (MI_algo&COUNTING_ANN) 
    {   init_ANN_MI(npts, mx, my, k, nb_cores);          // for MI, there will be indeed 2*nb_cores
        create_kd_tree(x, npts, mx+my);     // old
        create_kd_tree_1(x, npts, mx);      // first variable is already pre-sorted
        create_kd_tree_2(y, npts, my);      // first variable is already pre-sorted
        // 2021-12-02: the three lines above can be threaded!
	}
	else    // NBG counting
    {   init_ANN(npts, n, k, nb_cores*2);   // 2021-12-13: check if 2*nb_cores is necessary or not 	
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
        my_arguments[core].k = k;       // thread independent
        my_arguments[core].x = x;       // thread independent
        my_arguments[core].y = y;       // thread independent
        my_arguments[core].indices_y = indices_y;       // thread independent
        my_arguments[core].ind_inv_y = ind_inv_y;       // thread independent
        my_arguments[core].unit_perm = unit_perm;       // thread independent
        // 2021-12-02: note that parameters are the same for the two algorithms
        if (MI_algo & MI_ALGO_1)
        {   if (MI_algo&COUNTING_ANN) 
            pthread_create(&thread_algo1[core], NULL, threaded_MI_algo1_ANN_func, (void *)&my_arguments[core]);
            else
            pthread_create(&thread_algo1[core], NULL, threaded_MI_algo1_NBG_func, (void *)&my_arguments[core]);
        }    
        if (MI_algo & MI_ALGO_2)
        {   if (MI_algo&COUNTING_ANN) 
            pthread_create(&thread_algo2[core], NULL, threaded_MI_algo2_ANN_func, (void *)&my_arguments[core]);
            else
            pthread_create(&thread_algo2[core], NULL, threaded_MI_algo2_NBG_func, (void *)&my_arguments[core]);
        }
    }

    for (core=0; core<nb_cores; core++)
    {   if (MI_algo & MI_ALGO_1) 
        {   pthread_join(thread_algo1[core], (void**)&my_outputs_algo1[core]);
            h1 += my_outputs_algo1[core]->I_sum;
            nb_errors_1 += my_outputs_algo1[core]->n_errors;
            nb_points_check1 += my_outputs_algo1[core]->n_eff;
            free(my_outputs_algo1[core]);
        }
        if (MI_algo & MI_ALGO_2) 
        {   pthread_join(thread_algo2[core], (void**)&my_outputs_algo2[core]);
            h2 += my_outputs_algo2[core]->I_sum;
            nb_errors_2 += my_outputs_algo2[core]->n_errors;
            nb_points_check2 += my_outputs_algo2[core]->n_eff;
            free(my_outputs_algo2[core]);
        }          
    }
    
    if (!(MI_algo & MI_ALGO_1)) h1=my_NAN;
    if (!(MI_algo & MI_ALGO_2)) h2=my_NAN;

    if (lib_verbosity>1)
    {   if ( (MI_algo & MI_ALGO_1) && (nb_points_check1 != npts) )
            print_warning(fname, "nb pts not conserved!");
        if ( (MI_algo & MI_ALGO_2) && (nb_points_check2 != npts) )
            print_warning(fname, "nb pts not conserved!");
    }

    nb_errors_local = nb_errors_1 + nb_errors_2; // for how many points we haven't found a k-th closest neighbour

    if (lib_warning_level>=1)   // physical checks, treated as a warning
 	{   if (nb_errors_1>0) 
 	    {   sprintf(message, "%d errors encountered in algo 1: be carefull, the result may not be valid !", nb_errors_1);
 	        print_warning(fname, message);
 	    }
        if (nb_errors_2>0) 
        {   sprintf(message, "%d errors encountered in algo 2: be carefull, the result may not be valid !", nb_errors_2);
            print_warning(fname, message);
        }
    }
	if (npts<=nb_errors_local) // test re-organized on 2019-01-21
    {   sprintf(message, "npts=%d points in dataset >= %d errors. aborting...\n", npts, nb_errors);
        print_error(fname, message);
    }
    else
    {   h1 = h1/(double)(npts-nb_errors_1); /* normalisation de l'esperance */
        h2 = h2/(double)(npts-nb_errors_2); /* normalisation de l'esperance */
	
        h1 += gsl_sf_psi_int (k) + gsl_sf_psi_int (npts-nb_errors_1);
        h2 += gsl_sf_psi_int (k) + gsl_sf_psi_int (npts-nb_errors_2) - (double)1.0/(double)k;
	
        *I1 = h1;
        *I2 = h2;
    }
    
	/* then we restore initial pointers (before orderings by QuickSort) */
	for (d=0; d<n; d++)
	{	y_tmp = x+(d*npts);
		for (i=0; i<npts; i++) toto[i]  = y_tmp[ind_inv_x[i]];
		for (i=0; i<npts; i++) y_tmp[i] = toto[i];
	}
	
	/* free pointers de taille n=dimension de l'espace : */
    free(indices_x); free(ind_inv_x);
	free(indices_y); free(ind_inv_y);
	free(y);  free(toto);
    if (MI_algo&COUNTING_ANN) free_ANN_MI(nb_cores);
    else {                    free_ANN(2*nb_cores); // 2022-03-19: added factor 2
	                          free(unit_perm);
	     }
	return(nb_errors_local);
} /* end of function "compute_mutual_information_2xnd_ann_thread" */


