/*
 *  entropy_ann_single_MI.c
 *
 *  this file contains the "engine" function that computes MI
 *  counting is performed based on either
 *  - an N.B.G. algorithm (legacy)
 *  - the ANN library : http://www.cs.umd.edu/~mount/ANN/
 *
 *  this file is included in "entropy_ann.c"
 *
 *  Created by Nicolas Garnier on 2021-12-10
 *  Copyright 2012-2021 ENS-Lyon - CNRS. All rights reserved.
 *
 *  these functions are wrapped within the Cython code,
 *  where they can be called by setting a flag in the algo choice
 *  ->  (|=0x0100) for ANN counting
 *
 *  2019-01-29 : forked from entropy_ann.c
 *  2019-01-31 : functions working fine
 *  2020-02-27 : new management of nb_errors
 *  2021-12-02 : thread management (forked out in new file)
 *  2021-12-09 : split from "entropy_ann_new.c" for MI only
 *  2021-12-10 : new unique function with both counting algorithms, and timings
 *  2021-12-13 : added several tests (activated with #defines)
 */

#include <stdlib.h>
#include <stdio.h>              // for printf
#include <string.h>             // for malloc
#include <math.h>               // for log 
#include <gsl/gsl_sf.h>         // for psi digamma function

#include "library_commons.h"    // definitions of nb_errors, and stds
#include "library_matlab.h"     // compilation for Matlab
#include "ANN_wrapper.h"        // for ANN library (in C++)
#include "entropy_ann.h"
#include "nns_count.h"          // counting functions (2019-01-23)
#include "math_tools.h"

#define noDEBUG
#define DEBUG_N 37
#define N_test 100000
#define noDEBUG_EXPORT

#define noTIMING
#define noTIMING_COUNTING

#ifdef TIMING
    #include "timings.h"
#else 
    #define tic()
    #define toc(x)    // 0.0
    #define tic_in()
    #define toc_in(x) // 0.0
#endif


/****************************************************************************************/
/* computes mutual information, using nearest neighbor statistics (Grassberger 2004)    */
/* this is an application of PRE 69 066138 (2004)									    */
/*                                                                                      */
/* this version is for (m+p)-dimentional systems							    		*/
/* and computes information redundency of 2 variables of dimension m and p		    	*/
/*																			            */
/* x   contains all the data, which is of size (m+p)*nx, (possibly huge)				*/
/* nx  is the number of points in time											        */
/* m   is the dimensionality of the first variable, p the one of the second variable	*/
/*																			            */
/* data is ordered like this (in case of m and p from some embedding):                  */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)				*/
/* components of first variables are first, and then are components of second variable	*/
/*																			            */
/* m and p can also be simply the dimensionality of x and y (no embedding)              */
/* or the product of initial dimensionality by embedding dimension                      */
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
/* 2012-03-14 : test : added factor (n-1) in front of psi(nx) for correct               */
/*               normalisation according to Kraskov formula (23)       THEN REMOVED !   */
/*																			            */
/* to-do : test sur erreur grave lorsque nb_error > nb_voisin						    */
/* 2019-01-28 : algo 2 is back!                                                         */
/* 2019-01-30 : using a set of new functions (using ANN) to count the neighbors         */
/*              results looks OK-ish, but there is a severe decrease of performance     */
/*              compared to N.G. trick with pre-sorting                                 */
/* 2019-01-31 : => pre-sorting incorporated on new functions => factor 2 in speed!      */
/* 2019-12-17 : correction for algo 1													*/
/* 2020-02-27 : using global variable "nb_errors_local" to return nb of errors			*/
/* 2021-12-10 : 2 functions merged into 1 (counting by NBG or by ANN)                   */
/****************************************************************************************/
int compute_mutual_information_direct_ann(double *x, int nx, int m, int p, int k, double *I1, double *I2)
{	
	register int d, i, j;
	int     *indices_x, *ind_inv_x, *indices_y, *ind_inv_y;
	int     n, nb_errors_1=0, nb_errors_2=0;
    int     n_x1, n_y1;  /* for algo 1 */
    int     n_x2, n_y2;  /* for algo 2 */
    double	epsilon=0;	/* ancien rayon de la boule (valeur au pas précédent) */
    double *epsilon_z;  /* for algorithm 2 */
    double epsilon_1, epsilon_2;
	double h1=0.0, h2=0.0;
	double *y, *toto;
	double *yi; 
	double *y_tmp;          // tmp pointer (will stay pure pointer)
	int    *unit_perm;
#ifdef TIMING_COUNTING	
	int     *test_i; // 2021-12-13 testing speed 
	double  *test_c;
	double  t_count_in=0, t_count_ex=0, t_count_marginal=0; 
#endif	
#ifdef DEBUG	
    int     n_corrections_1=0; // 2021-12-10 to count the number of corrections in algo 1
	double  eps_tmp;
#endif	
#ifdef TIMING
    double time_alloc=0, time_sort=0, time_init_ANN=0, time_0_1=0, time_0_2=0, time_1=0, time_2=0;
#endif
			
    *I1 = my_NAN;
    *I2 = my_NAN;

	n = m+p; /* total dimensionality of data */
	if (n<2) return(printf("[compute_mutual_information_direct_ann] not enough dimensions in data (at least 2 required)\n"));
	
	tic_in();
    /* we allocate memory */
    indices_x = (int*)calloc(nx, sizeof(int));	    // permutation of indices to map x into x_sorted
	ind_inv_x = (int*)calloc(nx, sizeof(int));      // inverse permutation
	indices_y = (int*)calloc(nx, sizeof(int));      // only to count neighbors
    ind_inv_y = (int*)calloc(nx, sizeof(int));      // only to count neighbors
    unit_perm = (int*)calloc(nx, sizeof(int));      // unitary permutation (identity)
	for (j=0; j<nx; j++) unit_perm[j] = j;          // for NBG algo only
	
	yi		  = (double*)calloc(n, sizeof(double));
	epsilon_z = (double*)calloc(n, sizeof(double)); // for algo 2
	y         = (double*)calloc(nx*p, sizeof(double));
	toto      = (double*)calloc(nx, sizeof(double));  /* pure tmp variable  */
    toc_in(&time_alloc);
    
    tic_in();
	// x corresponds to the favored dimension d=0 :
	for (i=0; i<nx; i++) indices_x[i]=i;	
	QuickSort_double(x, indices_x, 0, nx-1);	// we sort the data, pointers x and indices_x are modified !
	for (i=0; i<nx; i++) ind_inv_x[indices_x[i]]=i;
	                    // so we have x_sorted[ind_inv_x[i]] = x_unsorted[i]
	                    // and        x_unsorted[ind[i]] = x_sorted[i]
	// we arrange all other dimensions accordingly : 
	for (d=1; d<m+p; d++)
	{	y_tmp = x+(d*nx);
		for (i=0; i<nx; i++) toto[i]  = y_tmp[indices_x[i]];
		for (i=0; i<nx; i++) y_tmp[i] = toto[i];
	}
	
	/* to save some time when counting, we order first dimension of second variable y now :	*/
	memcpy(y, x+m*nx, nx*p*sizeof(double));
	for (j=0; j<nx; j++) indices_y[j] = j;
	QuickSort_double(y, indices_y, 0, nx-1);
		// 2011/07/14 : we only sort direction m which is the first direction of the second vector
	for (j=0; j<nx; j++) ind_inv_y[indices_y[j]] = j;

 // 2012-12-10: test: not arranging other dimensions for ANN-counting makes it a failure!
	if (MI_algo&COUNTING_ANN) 
    {  for (d=1; d<p; d++) // we arrange all other dimensions accordingly :
        {   y_tmp = y+d*nx;
            for (i=0; i<nx; i++) toto[i]  = y_tmp[indices_y[i]];
            for (i=0; i<nx; i++) y_tmp[i] = toto[i];
        }
    } 
    toc_in(&time_sort);

    tic_in();
    if (MI_algo&COUNTING_ANN) 
    {   init_ANN_MI(nx, m, p, k, SINGLE_TH);
        create_kd_tree(x, nx, m+p); // old
        create_kd_tree_1(x, nx, m); // first variable is already pre-sorted
        create_kd_tree_2(y, nx, p);
    }
    else
    {   init_ANN(nx, n, k, SINGLE_TH); 	
        create_kd_tree(x, nx, n);
    }
	toc_in(&time_init_ANN);
	nb_errors_local=0;    /* for how many points we haven't found a k-th closest neighbour */
    
#ifdef DEBUG_EXPORT
    FILE *fic;
    fic = fopen("MI_ANN.txt", "wt");
#endif 
    
    if (MI_algo & MI_ALGO_1) for (i=0; i<nx; i++)
	{	for (d=0; d<n; d++) yi[d] = x[d*nx + i];	/* composantes zi de z */
#ifdef DEBUG_EXPORT
        fprintf(fic, "%3d, center: ", i);
        for (d=0; d<n; d++) fprintf(fic, "%2f ", yi[d]); // center point
#endif 
        tic_in();
        if (MI_algo&COUNTING_ANN) 
        {   // ANN counting
            ANN_marginal_distances_ex(yi, n, k, epsilon_z, 0);  // 2019-12-17, for the correction below 
                                                                // 2021-12-01 pthread
            epsilon=my_max(epsilon_z, n);   // 2019-12-17: tested identical to ANN_find_distance_in(i, n, k)
                                            // 2021-12-13: speed tests : almost no difference with _distance_in   
            
#ifdef DEBUG // 2021-12-12 - debug 
            if (i==DEBUG_N)
            {   printf("i=37 (k=%d) - epsilon_z = [ ", k);
                for (d=0; d<n; d++) printf("%f  ", epsilon_z[d]);
                printf("] => epsilon = %f\n", epsilon);
                printf("now using ANN_find_distance_in :\n");
                for (int q=1; q<=k; q++)
                {   eps_tmp = ANN_find_distance_in(i, n, q, 0);
                    printf("k=%2d - epsilon = %f\n", q, eps_tmp);
                }
                printf("now using ANN_marginal_distances_ex :\n");
                for (int q=1; q<=k; q++)
                {   eps_tmp = ANN_marginal_distances_ex(yi, n, q, epsilon_z, 0);
                    printf("k=%2d - epsilon_z =[ ", q);
                    for (d=0; d<n; d++) printf("%f  ", epsilon_z[d]);
                    printf("] => epsilon = %f", my_max(epsilon_z, n));
                    printf("\t eps_x = %f / eps_y = %f\n", my_max(epsilon_z, m), my_max(epsilon_z+m, p));
                }
            }
#endif             
#ifdef TIMING_COUNTING	// 2021-12-13: speed test
            if (i==DEBUG_N)
            {   toc_in(&time_0_1); // stop counter
                test_i = calloc(N_test, sizeof(int));
                test_c = calloc(N_test*n, sizeof(double));
                for (int q=0; q<N_test; q++) 
                {   test_i[q] = rand()%nx;
                    for (d=0; d<n; d++) test_c[q*n + d] = x[d*nx + test_i[q]];
                }
                tic_in();
                for (int q=0; q<N_test; q++)
                    {   ANN_marginal_distances_ex(test_c+q*n, n, k, epsilon_z, 0);
                    }
                toc_in(&t_count_marginal);
                tic_in();
                for (int q=0; q<N_test; q++)
                    {   epsilon=ANN_find_distance_ex(test_c+q*n, n, k, 0);
                    }
                toc_in(&t_count_ex);
                tic_in();
                for (int q=0; q<N_test; q++)
                    {   epsilon=ANN_find_distance_in(test_i[q], n, k, 0);
                    }
                toc_in(&t_count_in);
                printf("\t\ttiming of counting algos:\t marginal ex: %f\tdist ex: %f\tdist in: %f\n", 
                                    t_count_marginal, t_count_ex, t_count_in);
                tic_in(); // restart counter
            }
#endif            
        }
        else 
        {   // NBG counting
            epsilon = ANN_find_distance_in(i, n, k, 0); // 2019-01-22, 2021-12-01
        }
        toc_in(&time_0_1);
#ifdef DEBUG_EXPORT
        fprintf(fic, "\teps: ");
        fprintf(fic, "%2f ", epsilon);  // epsilon
#endif 
        
        tic_in();
        if (epsilon>0)
        {   
            if (MI_algo&COUNTING_ANN) // should we use the version with ANN counting?
            {   n_x1 = ANN_count_nearest_neighbors_nd_tree1(yi,   epsilon, 0);
        	    n_y1 = ANN_count_nearest_neighbors_nd_tree2(yi+m, epsilon, 0);
            // 2019-12-17, below is a correction because we want < and not <= for algo 1:
        	    if (my_max(epsilon_z, m)==epsilon) // the k-th neighbor is at the boundary in the x dimension
        	    {	n_x1 -=1; // so it was counted wrongly
#ifdef DEBUG
        	        n_corrections_1 ++;
#endif        	        
        	    }
        	    else n_y1 -=1; // it was in the y dimensions
        	// 2019-12-17: both case are mutualy exclusive: tested OK
        	// 2019-12-17: note this correction is imperfect, because it uses only the k-th neighbor
        	// 				and maybe there are other points at the boundary... (?)
        	// 2021-12-10: the results with this corrections are exactly the same as from the NBG algo
        	}
        	else // N.B.G. algo
        	{   n_x1 = count_nearest_neighbors_nd_algo1(x, nx, m,           i,  unit_perm, yi,   epsilon);
			    n_y1 = count_nearest_neighbors_nd_algo1(y, nx, p, ind_inv_y[i], indices_y, yi+m, epsilon);
            }

#ifdef DEBUG_EXPORT
            fprintf(fic, "\tnx_x,y (1): ");
            fprintf(fic, "%2d %2d ", n_x1, n_y1); // counting
#endif   
            h1 = h1 - gsl_sf_psi_int(n_x1+1) - gsl_sf_psi_int(n_y1+1);
		}
        else // (epsilon==0)
        {   nb_errors_1++;
        }
        toc_in(&time_1);
#ifdef DEBUG_EXPORT
        fprintf(fic, "\n");
#endif 
	} // algo_1, end of loop on i
    else h1=my_NAN;
 
    if (MI_algo & MI_ALGO_2) for (i=0; i<nx; i++)
    {   for (d=0; d<n; d++) yi[d] = x[d*nx + i];    /* composantes zi de z */

#ifdef DEBUG_EXPORT
        fprintf(fic, "%3d, center: ", i);
        for (d=0; d<n; d++) fprintf(fic, "%2f ", yi[d]); // center point
#endif    
        tic_in();
        ANN_marginal_distances_ex(yi, n, k, epsilon_z, 0); // new 2019-01-28 // 2021-12-01 pthread
//        epsilon   = my_max(epsilon_z, n);
        epsilon_1 = my_max(epsilon_z,   m); // 2019-01-30
        epsilon_2 = my_max(epsilon_z+m, p);
        toc_in(&time_0_2);
        
#ifdef DEBUG_EXPORT
        fprintf(fic, "\teps_z: ");
        for (d=0; d<n; d++) fprintf(fic, "%2f ", epsilon_z[d]); // epsilon
#endif
        tic_in();
        if (epsilon_1*epsilon_2>0) // let's be strict
        {   if (MI_algo&COUNTING_ANN)
            {   n_x2 = ANN_count_nearest_neighbors_nd_tree1(yi,   epsilon_1, 0);
                n_y2 = ANN_count_nearest_neighbors_nd_tree2(yi+m, epsilon_2, 0);
            }
            else
            {   n_x2 = count_nearest_neighbors_nd_algo2(x, nx, m,           i,  unit_perm, yi, epsilon_1);          
                n_y2 = count_nearest_neighbors_nd_algo2(y, nx, p, ind_inv_y[i], indices_y, yi+m, epsilon_2);
            }
            
            h2 = h2 - gsl_sf_psi_int(n_x2) - gsl_sf_psi_int(n_y2);
            
#ifdef DEBUG_EXPORT
            fprintf(fic, "\tnx_x,y (2): ");
            fprintf(fic, "%2d %2d ", n_x2, n_y2); // counting
#endif                        
        }
        
        else // (epsilon==0)
        {   nb_errors_2++;
        }

        toc_in(&time_2);
#ifdef DEBUG_EXPORT
        fprintf(fic, "\n");
#endif     
        
    } // algo_2, end of loop on i
    else h2=my_NAN;

#ifdef DEBUG_EXPORT
    fclose(fic);
#endif 

    nb_errors_local = nb_errors_1 + nb_errors_2;
    
#ifdef DEBUG	
 	if (nb_errors_1>0) printf("[compute_partial_MI_direct_ann] %d errors encountered in algo 1: be carefull, the result may not be valid !\n", nb_errors_1);
    if (nb_errors_2>0) printf("[compute_partial_MI_direct_ann] %d errors encountered in algo 2: be carefull, the result may not be valid !\n", nb_errors_2);
#endif
	if (nx<=nb_errors_local) // test re-organized on 2019-01-21
    {
        printf("[compute_mutual_information_direct_ann] pb : nx=%d points in dataset, %d errors. aborting...\n", nx, nb_errors);
    }
    else
    {   h1 = h1/(double)(nx-nb_errors_1); /* normalisation de l'esperance */
        h2 = h2/(double)(nx-nb_errors_2); /* normalisation de l'esperance */
	
        h1 += gsl_sf_psi_int (k) + gsl_sf_psi_int (nx-nb_errors_1);
        h2 += gsl_sf_psi_int (k) + gsl_sf_psi_int (nx-nb_errors_2) - (double)1.0/(double)k;
	
        *I1 = h1;
        *I2 = h2;
    }
    
	/* then we restore initial pointers (before orderings by QuickSort) */
	for (d=0; d<n; d++)
	{	y_tmp = x+(d*nx);
		for (i=0; i<nx; i++) toto[i]  = y_tmp[ind_inv_x[i]];
		for (i=0; i<nx; i++) y_tmp[i] = toto[i];
	}
	
	/* free pointers de taille n=dimension de l'espace : */
    free(indices_x); free(ind_inv_x);
	free(indices_y); free(ind_inv_y);
	free(y);  free(toto);
	free(yi); free(epsilon_z);
	free(unit_perm);
	
	if (MI_algo&COUNTING_ANN) free_ANN_MI(SINGLE_TH);
	else                      free_ANN(SINGLE_TH);  
	
#ifdef TIMING
    printf("[compute_mutual_information_direct_ann] timings:\n\t\t\t%f for alloc., %f for sorting, %f for ANN init,\n\t\t\t%f for k-search 1, %f for algo 1 counting,\n\t\t\t%f for k-search 2, %f for algo 2 counting\n", time_alloc, time_sort, time_init_ANN, time_0_1, time_1, time_0_2, time_2);
#endif
#ifdef DEBUG
	if (MI_algo&COUNTING_ANN) printf("\t\t%d corrections along x.\n", n_corrections_1);
#endif 	
	//	fclose(logfile);
	return(nb_errors_local);
} /* end of function "compute_mutual_information_direct_ann" */


