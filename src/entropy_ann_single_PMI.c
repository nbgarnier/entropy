/*
 *  entropy_ann_single_PMI.c
 *
 *  this file contains the unified "engine" function that computes PMI
 *  counting is performed based on either
 *  - an N.B.G. algorithm (legacy)
 *  - the ANN library : http://www.cs.umd.edu/~mount/ANN/
 *
 *  this file is included in "entropy_ann.c"
 *
 *  Created by Nicolas Garnier on 2021/12/14.
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
 *  2021-12-09 : split from "entropy_ann_new.c" for PMI only
 *  2021-12-14 : new single function with both counting algorithms
 */

#include <string.h>     // for malloc
#include <math.h>       // for log 
#include <gsl/gsl_sf.h> // for psi digamma function

#include "library_commons.h"    // definitions of nb_errors, and stds
#include "library_matlab.h"     // compilation for Matlab
#include "ANN_wrapper.h"        // for ANN library (in C++)
#include "entropy_ann.h"
#include "nns_count.h"          // counting functions (2019-01-23)
#include "math_tools.h"


/***************************************************************************************
 * computes partial mutual information, as defined by Frenzel and Pompe
 * using nearest neighbor statistics (Grassberger 2004)
 * this version is for (m+p+q)-dimensional systems	 
 *
 * (ordering of data is a bit strange, please use the wrapper instead)							
 *
 * for the definition, and the estimator, see the article from Frenzel, Pompe 
 *		 PRL 99, 204101 (2007)
 *		 "Partial Mutual Information for Coupling Analysis of Multivariate Time Series"
 *
 * I(X,Y|Z) = part of the MI I(X,Y) which is not in Z
 *
 * x      contains all the data, which is of size (m+p+q)*nx, that is, huge
 * nx     is the number of points in time	
 * m      is the dimension of x (can be the nb of point in the past)
 * p      is the dimension of y
 * q      is the dimension of z, the conditioning variable
 * k      is the number of neighbors to consider
 *
 * data in x is ordered like this :										
 * x1(t=1)...x1(t=nx-1) x2(t=0) ... x2(t=nx-2) ... xn(t=0) ... xn(t=nx-2)
 * components of X are first, and then are components of conditioning variable Z,
 * and then are components of second variable Y                                         
 * If considering time embedding, dimensions are from future to past (causal)
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
 * 2021-12-14 : unified function for both ANN and NBG countings
 * 2021-12-14 : correction for ANN counting implemented, and checked OK
 ***************************************************************************************/
int compute_partial_MI_engine_ann(double *x, int nx, int m, int p, int q, int k, double *I1, double *I2)
{
	register int d, i, j;
	int	   *indices_x, *ind_inv_x, *indices_y, *ind_inv_y, *unit_perm=NULL;
	int    n, nb_errors_1=0, nb_errors_2=0;
    int     n_z, n_zy, n_xz;  /* for counting in both algo 1 and 2 */
    double	epsilon, epsilon_1, epsilon_2, epsilon_3;
    double *epsilon_z; /* pour algorithme 2 */
	double h1=0.0, h2=0.0;
	double *y, *toto;   // will be allocated
	double *yi; 
	double *y_tmp;      // pure pointer
    
    *I1=my_NAN; *I2=my_NAN;

	n = m+p+q; /* total dimensionality of data */
	if (n<3) return(printf("[compute_partial_MI_engine_ann] : not enough dimensions in data (at least 3 required)\n"));
	
    /* we allocate memory */
	indices_x = (int*)calloc(nx, sizeof(int));	/* permutation des indices pour passer de x à x sorted */
	ind_inv_x = (int*)calloc(nx, sizeof(int));  /* permutation inverse */
    indices_y = (int*)calloc(nx, sizeof(int));  /* only to count neighbors    */
    ind_inv_y = (int*)calloc(nx, sizeof(int));  /* only to count neighbors    */
    y         = (double*)calloc(nx*(q+p), sizeof(double));      // (z,y)
	toto      = (double*)calloc(nx, sizeof(double));            // pure tmp
    yi        = (double*)calloc(n, sizeof(double));
    epsilon_z = (double*)calloc(n, sizeof(double));
    
    if ( (toto==NULL) || (ind_inv_y==NULL) ) return(printf("[compute_partial_MI_engine_ann] : alloc error\n"));
	
    for (i=0; i<nx; i++) indices_x[i]=i;
	QuickSort_double(x, indices_x, 0, nx-1);		/* we sort the data, pointers x and indices_x are modified ! */
	for (i=0; i<nx; i++) ind_inv_x[indices_x[i]]=i;	
	    /* so we have x_sorted[ind_inv_x[i]] = x_unsorted[i] */
	    /* and        x_unsorted[ind[i]] = x_sorted[i]		*/
	for (d=1; d<m+p+q; d++) /* we arrange all other dimensions accordingly : */
	{	y_tmp = x+(d*nx);
		for (i=0; i<nx; i++) toto[i]  = y_tmp[indices_x[i]];
		for (i=0; i<nx; i++) y_tmp[i] = toto[i];
	}

	/* to save some time when counting, we order the first dimension of (z,y) now :	*/
	memcpy(y, x+m*nx, nx*(q+p)*sizeof(double));
	for (j=0; j<nx; j++) indices_y[j] = j;
	QuickSort_double(y, indices_y, 0, nx-1);   // 2012/01/03 : we only sort first direction
                                            //  of both (z,y) and (z) (without x)
	for (j=0; j<nx; j++) ind_inv_y[indices_y[j]] = j;
	
	if (MI_algo&COUNTING_ANN)
    {	for (d=1; d<(q+p); d++) /* we arrange all other dimensions accordingly : */
        {   y_tmp = y+d*nx;
            for (i=0; i<nx; i++) toto[i]  = y_tmp[indices_y[i]];
            for (i=0; i<nx; i++) y_tmp[i] = toto[i];
        }
    }
    else // for NBG algo only
    {   unit_perm = (int*)calloc(nx, sizeof(int));
        for (j=0; j<nx; j++) unit_perm[j] = j;
    }

    if (MI_algo&COUNTING_ANN)
    {   init_ANN_PMI(nx, m, p, q, k, SINGLE_TH);
        create_kd_tree  (x, nx, m+q+p);  // full tree
        create_kd_tree_1(x, nx, m+q);    // XZ tree
        create_kd_tree_2(y, nx, q);      // Z tree
        create_kd_tree_3(y, nx, q+p);    // ZY tree
    }
    else
    {   init_ANN(nx, n, k, SINGLE_TH);
        create_kd_tree(x, nx, n);
	}
    
    nb_errors_local=0;
	
    if (MI_algo & MI_ALGO_1) for (i=0; i<nx; i++)
	{	for (d=0; d<n; d++) yi[d] = x[d*nx + i];	/* composantes zi de z */
		
		if (MI_algo&COUNTING_ANN)
        {   ANN_marginal_distances_ex(yi, n, k, epsilon_z, 0);  
            epsilon=my_max(epsilon_z, n);
        }
        else
        {   epsilon = ANN_find_distance_in(i, n, k, 0); // 2019-01-22 // 2021-12-01
        }
        
		if (epsilon>0)
		{
		    if (MI_algo&COUNTING_ANN) // should we use the version with ANN counting?
		    {   n_xz = ANN_count_nearest_neighbors_nd_tree1(yi,   epsilon, 0);
                n_z  = ANN_count_nearest_neighbors_nd_tree2(yi+m, epsilon, 0);
                n_zy = ANN_count_nearest_neighbors_nd_tree3(yi+m, epsilon, 0);

                if (my_max(epsilon_z+m, q)==epsilon)    // k-th nn is at the boundary along (z)
                    n_z -=1;
                if (my_max(epsilon_z, m+q)==epsilon)    // k-th nn is at the boundary along (xz)
                    n_xz -=1;
                if (my_max(epsilon_z+m, q+p)==epsilon)  // k-th nn is at the boundary along (zy)
                    n_zy -=1;
            }
            else
            {	n_xz = count_nearest_neighbors_nd_algo1(x, nx, m+q, i,            unit_perm,     yi, epsilon);
                n_z  = count_nearest_neighbors_nd_algo1(y, nx, q,   ind_inv_y[i], indices_y,   yi+m, epsilon);
			    n_zy = count_nearest_neighbors_nd_algo1(y, nx, q+p, ind_inv_y[i], indices_y,   yi+m, epsilon);
			}
            
			h1 = h1 + gsl_sf_psi_int(n_z+1) - gsl_sf_psi_int(n_xz+1) - gsl_sf_psi_int(n_zy+1);
		}
        else nb_errors_1++; // (epsilon<=0)
	} // algo_1, end of loop on i
    else h1=my_NAN;


    if (MI_algo & MI_ALGO_2) for (i=0; i<nx; i++)
    {    for (d=0; d<n; d++) yi[d] = x[d*nx + i];    /* composantes zi de z */
        
        ANN_marginal_distances_ex(yi, n, k, epsilon_z, 0); // new 2019. // 2021-12-01 pthread
//        epsilon   = my_max(epsilon_z,   m+q+p);
        epsilon_1 = my_max(epsilon_z,   m+q); // 2019-01-30
        epsilon_2 = my_max(epsilon_z+m, q);
        epsilon_3 = my_max(epsilon_z+m, q+p);

        if ((epsilon_1*epsilon_2*epsilon_3)>0) // more serious then (epsilon>0)
        {   if (MI_algo&COUNTING_ANN)
            {   n_xz = ANN_count_nearest_neighbors_nd_tree1(yi,   epsilon_1, 0);
                n_z  = ANN_count_nearest_neighbors_nd_tree2(yi+m, epsilon_2, 0);
                n_zy = ANN_count_nearest_neighbors_nd_tree3(yi+m, epsilon_3, 0);
            }
            else
            {   n_xz = count_nearest_neighbors_nd_algo2(x, nx, m+q, i,            unit_perm, yi,   epsilon_1);
                n_z  = count_nearest_neighbors_nd_algo2(y, nx, q,   ind_inv_y[i], indices_y, yi+m, epsilon_2);
                n_zy = count_nearest_neighbors_nd_algo2(y, nx, q+p, ind_inv_y[i], indices_y, yi+m, epsilon_3);
            }
            h2 = h2 + gsl_sf_psi_int(n_z) - gsl_sf_psi_int(n_xz) - gsl_sf_psi_int(n_zy);
        }
        else nb_errors_2++; // (epsilon<=0)
    } // algo_2, end of loop on i
    else h2=my_NAN;
    
    
    nb_errors_local = nb_errors_1 + nb_errors_2; // 2020-02-27: errors may be counted twice if both algos are used
#ifdef DEBUG	
    if (nb_errors_1>0) printf("[compute_partial_MI_direct_nns] %d errors encountered in algo 1: be carefull, the result may not be valid !\n", nb_errors_1);
    if (nb_errors_2>0) printf("[compute_partial_MI_direct_nns] %d errors encountered in algo 2: be carefull, the result may not be valid !\n", nb_errors_2);
#endif
	if (nx<=nb_errors_local)
        printf("[compute_partial_MI_direct_nns] pb : nx=%d points in dataset, %d errors. aborting...\n",
               nx, nb_errors_local);
    else
    {   h1 = h1/(double)(nx-nb_errors_1); /* normalisation de l'esperance */
        h2 = h2/(double)(nx-nb_errors_2); /* normalisation de l'esperance */
	
        /* ci-après, normalisation finale : */
        h1 = h1 + gsl_sf_psi_int(k);
        h2 = h2 + gsl_sf_psi_int(k); // - (double)2.0/(double)k;
                                     // 2 = 3-1, with 3 marginal directions in the combinaison.
                // 2019-02-01 : removed a factor, which should not have been there!!! XII.259
        *I1 = h1;
        *I2 = h2;
    }
    
	/* then we restore initial pointers (as they were before being ordered by QuickSort) */
    // 2025-03-17: bug (found by Ewen Frogé) corrected
	for (d=0; d<n; d++) // 2021-12-14: code below looks wrong!!! to check!!! 2025-03-17: indeed!
	{	y_tmp = x+(d*nx);
		for (i=0; i<nx; i++) toto[i]  = y_tmp[ind_inv_x[i]];
		for (i=0; i<nx; i++) y_tmp[i] = toto[i];	
	}	

	free(indices_x); free(ind_inv_x);
    free(indices_y); free(ind_inv_y);
	free(y); free(toto);
    free(yi);
	free(epsilon_z);
	
	if (MI_algo&COUNTING_ANN) free_ANN_PMI(SINGLE_TH);
	else                      
	{                         free_ANN(SINGLE_TH);
	                          free(unit_perm);
	}
	
	return(nb_errors_local);
} /* end of function "compute_partial_MI_engine_ann" */

