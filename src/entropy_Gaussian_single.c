/*
 *  entropy_Gaussian_single.c
 *  
 *  to compute Shannon entropy assuming Gaussian statistics 
 *
 *  Created by Nicolas Garnier on 2022/10/10.
 *  Copyright 2012-2022 ENS-Lyon - CNRS. All rights reserved.
 *
 */

#include <math.h>               // for fabs and log
#include <string.h>
#include <gsl/gsl_statistics_double.h>

#include "library_commons.h"    // definitions of nb_errors, and stds
#include "library_matlab.h"     // compilation for Matlab
#include "math_tools.h"

#ifndef M_PI
// #define M_PI (3.14159265358979323846)
#define M_PI (3.14159265358979323846264338327950288)
#endif

#define noDEBUG	    // for debug information, replace "noDEBUG" by "DEBUG"
#define noDEBUG_EXPORT
#define LOOK 17 	// for debug also (of which point(s) will we save the data ?)



/****************************************************************************************/
/* computes Shannon entropy, using Gaussian statistics assumption                       */
/*                                                                                      */
/* this version is for n-dimentional systems                                            */
/*                                                                                      */
/* this version does not support embedding per se, but can be used by a wrapper which   */
/* embedds the data (the function "compute_entropy_Gaussian()" being exactly this)      */
/*                                                                                      */
/* x   contains all the data, which is of size n*nx                                     */
/* nx  is the number of points in time                                                  */
/* n   is the dimensionality                                                            */
/*                                                                                      */
/* data is ordered like this :                                                          */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)               */
/*                                                                                      */
/* 2012-02-27, fork from "compute_entropy_ann"                                          */
/****************************************************************************************/
double compute_entropy_nd_Gaussian(double *x, int npts, int n)
{	register int i,j;
	double h=my_NAN;
	double *M, det;
    
    M=(double*)calloc(n*n, sizeof(double));
    nb_errors_local=0;
    
    for (i=0; i<n; i++)
    {   M[i+n*i] = gsl_stats_variance(x+i*npts, 1, npts);
        for (j=i+1; j<n; j++)
        {   M[i+n*j] = gsl_stats_covariance(x+i*npts, 1, x+j*npts, 1, npts);
            M[j+n*i] = M[i+n*j];   // symetrical matrix
        }
    }  
    det = determinant(M, n);
//    printf("\tdet = %f\n", det);

    if (det>0.0) h = (double)n/2.*(1. + log(2.*M_PI)) + 1./2.*log(det);
    else printf("[compute_entropy_nd_Gaussian] negative det : %f\n", det);
    
	last_npts_eff_local = npts;
	free(M);
	return(h);
} /* end of function "compute_entropy_nd_ann" *******************************************/



/****************************************************************************************/
/* computes mutual information, for Gaussian statistics data						    */
/*                                                                                      */
/* this version is for (m+p)-dimentional systems							    		*/
/* and computes information redundency of 2 variables of dimension m and p		    	*/
/*																			            */
/* x     contains all the data, which is of size (m+p)*nx, (possibly huge)				*/
/* npts  is the number of points in time				    					        */
/* mx    is the dimensionality of the first variable,                                   */
/* my    is the dimensionality of the second variable                               	*/
/*																			            */
/* data is ordered like this (in case of mx and my from some embedding):                */
/* x1(t=0)...x1(t=nx-1) x2(t=0) ... x2(t=nx-1) ... xn(t=0) ... xn(t=nx-1)				*/
/* components of first variables are first, and then are components of second variable	*/
/*																			            */
/* mx and my can also be simply the dimensionality of x and y (no embedding)            */
/* or the product of initial dimensionality by embedding dimension                      */
/* For embedding, please use a wrapper                                                  */
/*																			            */
/* 2022-10-10 : new function                                                            */
/****************************************************************************************/
double compute_mutual_information_direct_Gaussian(double *x, int npts, int mx, int my)
{	register int i,j, m=mx+my;
	double h=my_NAN;
	double *Mx, *My, *M, detx, dety, det;
    
    Mx=(double*)calloc(mx*mx, sizeof(double));
    My=(double*)calloc(my*my, sizeof(double));
    M =(double*)calloc(m*m,   sizeof(double));
    nb_errors_local=0;
    
    for (i=0; i<m; i++)
    for (j=i; j<m; j++)
    {   M[i + m*j] = gsl_stats_covariance(x+i*npts, 1, x+j*npts, 1, npts);
        if (j>i)        // symetrical matrix
        {   M[j + m*i] = M[i + m*j];
        }
    }
    
    for (i=0; i<mx; i++) memcpy(Mx+i*mx, M+i*m, mx*sizeof(double));
    for (i=0; i<my; i++) memcpy(My+i*my, M+mx*m + mx + i*m, my*sizeof(double));

    detx = determinant(Mx, mx);
    dety = determinant(My, my);
    det  = determinant(M,  m);

    if (det>0.0) h = 1./2.*log(detx*dety/det);
    
	last_npts_eff_local = npts;
	free(Mx); free(My); free(M);
	return(h);
} /* end of function "compute_mutual_information_direct_Gaussian" *************************************/




/***************************************************************************************
 * computes partial mutual information, as defined by Frenzel and Pompe
 * for Gaussian statistics data			
 * this version is for (m+p+q)-dimensional systems	 
 *
 * (ordering of data is a bit strange, please use the wrapper instead)							
 *
 * I(X,Y|Z) = part of the MI I(X,Y) which is not in Z
 *
 * x      contains all the data, which is of size (m+p+q)*nx, that is, huge
 * npts   is the number of points in time	
 * mx     is the dimension of x (can be the nb of point in the past)
 * my     is the dimension of y
 * mz     is the dimension of z, the conditioning variable
 *
 * data in x is ordered like this :										
 * x1(t=1)...x1(t=nx-1) x2(t=0) ... x2(t=nx-2) ... xn(t=0) ... xn(t=nx-2)
 * components of X are first, and then are components of conditioning variable Z,
 * and then are components of second variable Y                                         
 * If considering time embedding, dimensions are from future to past (causal)
 *
 * 2021-12-14 : correction for ANN counting implemented, and checked OK
 ***************************************************************************************/
double compute_partial_MI_engine_Gaussian(double *x, int npts, int mx, int my, int mz)
{	register int i,j, m=mx+my+mz;
	double h=my_NAN;
	double *Mxz, *Mzy, *Mz, *M, detxz, detzy, detz, det;
    
    Mxz=(double*)calloc((mx+mz)*(mx+mz), sizeof(double));
    Mzy=(double*)calloc((mz+my)*(mz+my), sizeof(double));
    Mz =(double*)calloc( mz*mz,          sizeof(double));
    M  =(double*)calloc( m*m,            sizeof(double));
    nb_errors_local=0;
    
    for (i=0; i<m; i++)
    for (j=i; j<m; j++)
    {   M[i + m*j] = gsl_stats_covariance(x+i*npts, 1, x+j*npts, 1, npts);
        if (j>i)        // symetrical matrix
        {   M[j + m*i] = M[i + m*j];
        }
    }
    
    for (i=0; i<(mx+mz); i++) memcpy(Mxz+i*(mx+mz), M+i*m, (mx+mz)*sizeof(double));
    for (i=0; i<(mz+my); i++) memcpy(Mzy+i*(mz+my), M+mx*m + mx + i*m, (mz+my)*sizeof(double));
    for (i=0; i<mz;      i++) memcpy(Mz +i*(mz),    M+mx*m + mx + i*m, (mz)*sizeof(double));

    detxz = determinant(Mxz, mx+mz);
    detzy = determinant(Mzy, mz+my);
    detz  = determinant(Mz,  mz);
    det   = determinant(M,  m);
    
    if ((det>0.0) && (detz>0.0)) h = 1./2.*(log(detxz*detzy)-log(detz*det));
    
	last_npts_eff_local = npts;
	free(Mxz); free(Mzy); free(Mz); free(M);
	return(h);
} /* end of function "compute_mutual_information_direct_Gaussian" *************************************/



