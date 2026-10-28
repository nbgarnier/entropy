/*
 *  surrogates_PMI.c
 *
 *  this file contains the function to create surrogates (from a given dataset)
 *  especially tailored to satisfy the NULL hypothesis when computing the PMI 
 *  (so to estimate the bias of PMI/TE in the best possible way)
 *
 *  it implements the "local permutation" scheme defined in
 *  Jorge RUNGE - Proceedings of AISTATS, PMLR 84 pp938-947 (2018)
 *
 *  Created by Nicolas Garnier on 2026/10/27.
 *  Copyright 2012-2026 ENS-Lyon - CNRS. All rights reserved.
 *
 *  2026-10-27 : new function
 */

#include <string.h>     // for malloc
//#include <math.h>       // for log 
//#include <gsl/gsl_sf.h> // for psi digamma function


#include "library_commons.h"    // definitions of nb_errors, and stds
#include "library_matlab.h"     // compilation for Matlab
#include "ANN_wrapper.h"        // for ANN library (in C++)
#include "surrogates.h"         // for "init_surrogates_rng()"
#include "surrogates_PMI.h" // for "is_in_set()"
//#include "math_tools.h"


/***************************************************************************************
 * creates a set of surrogates for PMI bias estimation,
 * using nearest neighbor statistics
 *
 * for the definition of PMI, and the estimator, see the article from Frenzel, Pompe 
 *		 PRL 99, 204101 (2007)
 *		 "Partial Mutual Information for Coupling Analysis of Multivariate Time Series"
 *
 * for this surrogate generation method ("local permutation"), see Jorge RUNGE
 *       Proceedings of AISTATS, PMLR 84 pp938-947 (2018)
 *       "conditional independence testing"
 *
 * I(X,Y|Z) = part of the MI I(X,Y) which is not in Z
 * (ordering of data is a bit strange (see below), please use the wrapper instead)							
 *
 * x      contains all the data, which is of size (m+p+q)*nx, that is, huge
 * npts   is the number of points in time	
 * m      is the dimension of x (can be the nb of point in the past)
 * p      is the dimension of y
 * q      is the dimension of z, the conditioning variable
 * k_perm is the number of neighbors to consider
 * N_surrogates is the number of different surrogates to generate 
 * (because of the use of a k-d tree, it is much more efficient to generate all surrogates in one call)
 *
 * data in x is ordered like this :										
 * x1(t=1)...x1(t=nx-1) x2(t=0) ... x2(t=nx-2) ... xn(t=0) ... xn(t=nx-2)
 * components of X are first, and then are components of conditioning variable Z,
 * and then are components of second variable Y                                         
 *
 * this function uses subfunction(s) :								
 *	- ANN_get_nearest_neighbors (new function in ANN_wrapper.cpp 2026-10-27)							
 *
 * Some sanity tests are executed only ifdef DEBUG (compile with -dDEBUG to have the tests)
 *
 *											
 * 2026-10-27 : new file, new function
 ***************************************************************************************/
int surrogates_local_perm(double *x, int npts, int m, int p, int q, int k_perm, int N_surrogates, double *surrogates)
{	gsl_permutation *perm; 	// we need a permutation to ensure it is not alwayss the the same i that takes priority over other indices
	register int d, i, j, k, i_surrogate, i_permed;
    double	*z;  
	double  yi[q]; 
	int     used_ind[npts], n_used;
	int     *neighbors;     // to store the indices of neighbors of each central points
	int     n = m+p+q;      // total dimensionality of data. Note that Y can be completely omitted (so p=0) at no risk, 
							// but Y can also be provided (to keep compatibility with the call to compute_PMI_xxx() functions)
    
	init_surrogates_rng();	// if this hasn't been done already
	perm = gsl_permutation_alloc(npts);
        
	if (n<2) return(printf("[surrogates_local_perm] : not enough dimensions in data (at least 2 required for X and Z)\n"));
	
    /* we allocate memory */
    neighbors = (int*)calloc(npts*k_perm, sizeof(int));
    	
    z = x + m*npts; 				// the conditionning variable is the second one, after X, and before Y
    init_ANN(npts, q, k_perm, SINGLE_TH);
    create_kd_tree(z, npts, q);  	// Z tree only
            
    // search and store neighbors (in Z) of each point:
    for (i=0; i<npts; i++)
	{   for (d=0; d<q; d++) yi[d] = z[i];	// coordinates of z (central point)
	    (neighbors+k_perm*i)[0] = i;        // the first (nearest) neighbor is the central point itself
	    ANN_get_nearest_neighbors(yi, k_perm, neighbors+k_perm*i, SINGLE_TH);
	    // the other nearest neighbors are returned by the function above (new 2026-10-27)
	}
	free_ANN(SINGLE_TH);
	
    nb_errors_local=0;
	
	for (i_surrogate=0; i_surrogate<N_surrogates; i_surrogate++)
    {   for (i=0; i<npts; i++) 
		{	used_ind[i]=-1;   			// 0 is a valid index value, whereas -1 is not
			gsl_ran_shuffle(surrogates_rng, neighbors+k_perm*i, (size_t)k_perm, sizeof(int));
		}
		n_used = 0;						// nb of already used indices/points
		gsl_permutation_init(perm);		// additional precaution: we use a permutation to ensure no indice is favored
        gsl_ran_shuffle(surrogates_rng, perm->data, perm->size, sizeof(size_t));

        for (i=0; i<npts; i++)
	    {	i_permed = perm->data[i];	// we'll update randomly along the dataset's indices
			
			j = (neighbors+k_perm*i_permed)[0];
			k = 0;
			
			// we search for an indice j to use:
			while ( (is_in_set(j, used_ind, n_used)) && (k < k_perm) )
			{	k ++;
				j = (neighbors+k_perm*i_permed)[k];
			}

			// we use the indice j:
			for (d=0; d<m; d++)			// copy full x_j vector (size m) (at indice j)
				surrogates[i_permed +d*npts] = x[j +d*npts];
			for (d=m; d<n; d++)			// copy other dimensions (Y and Z) (if required, undecided yet)
				surrogates[i_permed +d*npts] = x[i_permed +d*npts];		// (they are unchanged)
			// and record it as used:
			used_ind[n_used] = j;
			n_used++;
        }
	} 

	free(neighbors);
	gsl_permutation_free(perm);
	return(nb_errors_local);
} /* end of function "compute_partial_MI_engine_ann" */


/****************************************************************************/
/* to test is an indice (i) is in a set (set) of maximal size (npts)		*/
/* 																			*/
/* 2026/10/28 : new function												*/
/****************************************************************************/
int is_in_set(int i, int *set, int npts)
{	register int j=0, in_set=0;
	
	while ( (!in_set) && (j<npts) )
	{	in_set = (i==set[j]);
		j++;
	}
	return(in_set);
}

