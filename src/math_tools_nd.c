/*
 *  math_tools_nd.c
 *  
 *  functions to operate on multi-variate data (n-dimensional)
 *
 *  Created by Nicolas Garnier on 13/08/10.
 *  Copyright 2010-2021 ENS-Lyon CNRS. All rights reserved.
 *
 * 2021-01-19: split from "math_tools.c" for clarity
 */

#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_matrix.h>      // for determinant of covariance
#include <gsl/gsl_permutation.h> // for determinant of covariance
#include <gsl/gsl_linalg.h>      // for determinant of covariance
#include <gsl/gsl_sf.h>          // for log and exp
#include "math_tools_nd.h"




/* we normalize a dataset with respect to its variance : */
/* and we return the std (in case we need it later)      */
/*												  */
/* 2011-12 : bug correction (loop from 0 instead of 1)   */
double normalize(double *x, int nx)
{	double  mn, /* mean */ 
			sd; /* standard deviation = sqrt(variance) */
	register int i;
	
	mn = gsl_stats_mean(x, 1 /* stride=1 */, nx);
	sd = gsl_stats_sd_m(x, 1 /* stride=1 */, nx, mn);	
	for (i=0; i<nx; i++)	x[i] = (x[i]-mn)/sd;
	
	return(sd);
}




/*********************************************************/
/* compute the covariance matrix of a set of vectors x   */
/* and put it in Sigma (which must be already allocated) */
/*                                                       */
/* used by function "determinant_covariance()"           */
/*                                                       */
/* 2013-02-20 first version                              */
/*********************************************************/
int compute_covariance_matrix(double *x, int m, int npts, double *Sigma)
{   register int i, j;
    
    for (i=0; i<m; i++)
    for (j=i; j<m; j++)
    {   if (i==j)
        {   Sigma[i + m*i] = gsl_stats_variance(x+i*npts, /* size_t stride */ 1, npts);
        }
        else
        {   Sigma[i + m*j] = gsl_stats_covariance(x+i*npts, /*const size_t stride1*/ 1, x+j*npts, /*const size_t stride2*/ 1, npts);
            // This GSL function computes the covariance of the datasets x and y which must both be of the same length npts.
            // covar = (1/(n - 1)) \sum_{i = 1}^{n} (x_i - \Hat x) (y_i - \Hat y)
            Sigma[j + m*i] = Sigma[i + m*j]; // covariance matrix is symetric ;-)
        }
    }
    return(0);
}

/*********************************************************/
/* compute the determinant of covariance matrix          */
/* of a set of vectors x and returns it                  */
/*                                                       */
/* 2013-02-20 first version                              */
/*********************************************************/
double determinant_covariance(double *x, int m, int npts)
{    double det=0.0;
     gsl_matrix *Sigma;
     gsl_permutation *permutation;
     int signum;
     
     Sigma       = gsl_matrix_alloc(m,m);    // http://www.gnu.org/software/gsl/manual/html_node/Matrix-allocation.html
     permutation = gsl_permutation_alloc(m); // http://www.gnu.org/software/gsl/manual/html_node/Permutation-allocation.html
     compute_covariance_matrix(x, m, npts, Sigma->data);
     
     gsl_linalg_LU_decomp (Sigma, permutation, &signum); // in-place LU decomposition of Sigma
     det = gsl_linalg_LU_det (Sigma, signum);
     
     gsl_permutation_free(permutation);
     gsl_matrix_free     (Sigma);
     
     return(det);
}

/*********************************************************/
/* normalize a matrix with respect to its covariance     */
/* and return the determinant of the covariance matrix   */
/*												  */
/* 2013-02-21 first version                              */
/*********************************************************/
double normalize_nd(double *x, int m, int npts)
{	double det=0.0;
    double  mn=0.0; /* mean */ 
//	double	sd; /* standard deviation = sqrt(variance) */
	register int i,j;
	
    det = determinant_covariance(x, m, npts);
//  printf("det : %f\n", det);
    if (det==0) 
    {   printf("[normalize_nd] Warning : determinant of covariance matrix is zero!\n"
                "No normalization has been performed.\n");
        return(0);
    }
    
    det = gsl_sf_exp( gsl_sf_log(det)/(double)(2*m));
    
    for (i=0; i<m; i++)
    {   mn = gsl_stats_mean(x+i*npts, 1 /* stride=1 */, npts);
        for (j=0; j<npts; j++)
            x[j + i*npts] = (x[j + i*npts]-mn)/det;
    }
	return(det);
}

