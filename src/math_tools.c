/*
 *  math_tools.c
 *  
 *
 *  Created by Nicolas Garnier on 13/08/10.
 *  Copyright 2010-2021 ENS-Lyon CNRS. All rights reserved.
 *
 * 2021-01-19: simplified by moving some functions to file "math_tools_nd.c"
 */

#include <stdlib.h>
#include <math.h>

#include "library_matlab.h"         // compilation for Matlab
#include "math_tools.h"



void QuickSort_int(int * xd, int * yd, int l, int r)
{	int i, j, x, y;
	int	k;
	
	i = l;  j = r;
	x = xd[ (int)((l+r)/2) ];
	while (i<j)
	{	while (xd[i] < x) { i ++; }
		while (x < xd[j]) { j --; }
		if (i <= j)
		{  	y = xd[i]; xd[i] = xd[j]; xd[j] = y;
			k = yd[i]; yd[i] = yd[j]; yd[j] = k;
			i ++;
			j --;
		}
	}
	
	if (l<j) QuickSort_int(xd, yd, l, j);
	if (i<r) QuickSort_int(xd, yd, i, r);
	
	return;
} 

/********************************************************/
/* this function operates on 2 arrays,					*/
/* it sorts the first one (floats)						*/
/* the second array is for the indices (permutations)	*/
/* to sort a pointer x of size N, call it with :        */
/*   QuickSort(x, indices, 0, N-1)						*/
/********************************************************/
void QuickSort_float(float *xd, int *yd, int l, int r)
{	register int  i, j, k;
	float         x, y;
	
	i = l;  j = r;
	x = xd[ (int)((l+r)/2) ];
	while (i<j)
	{	while (xd[i] < x) { i += 1; }
		while (x < xd[j]) { j -= 1; }
		if (i <= j)
		{  	y = xd[i]; xd[i] = xd[j]; xd[j] = y;
			k = yd[i]; yd[i] = yd[j]; yd[j] = k;
			i += 1;
			j -= 1;
		}
	}/*while*/
	if (l<j) QuickSort_float(xd, yd, l, j);
	if (i<r) QuickSort_float(xd, yd, i, r);
	return;
} 
/* end of the "QuickSort_f_i" function */

/********************************************************/
/* this function operates on 2 arrays,					*/
/* it sorts the first one (doubles)						*/
/* the second array is for the indices (permutations)	*/
/* to sort a pointer x of size N, call it with :        */
/*   QuickSort(x, indices, 0, N-0)						*/
/********************************************************/
void QuickSort_double(double *xd, int *yd, int l, int r)
{	register int i, j, k;
	double       x, y;
	
	i = l;  j = r;
	x = xd[ (int)((l+r)/2) ];
	while (i<j)
	{	while (xd[i] < x) { i += 1; }
		while (x < xd[j]) { j -= 1; }
		if (i <= j)
		{  	y = xd[i]; xd[i] = xd[j]; xd[j] = y;
			k = yd[i]; yd[i] = yd[j]; yd[j] = k;
			i += 1;
			j -= 1;
		}
	}/*while*/
	if (l<j) QuickSort_double(xd, yd, l, j);
	if (i<r) QuickSort_double(xd, yd, i, r);
	return;
} 
/* end of the "QuickSort_d_i" function */



/********************************************************************/
/* a function to "unsort" the data, and recover initial pointer		*/
/* Why ? after a Quicksort, pointers x and ind have been sorted !!! */
/* so we may have to "clean" them up, by making back pairs (xi,yi)	*/
/* if not, we have badly altered the dataset of z=(x,y)			    */
/********************************************************************/
int unsort_d_i(double *x, int *ind, int nx)
{	double  *x_tmp;
	int		*ind_inv;
	register int i;
	
	x_tmp   = (double*)calloc(nx, sizeof(double));
	ind_inv = (int*)   calloc(nx, sizeof(int)); 
	
	for (i=0; i<nx; i++) ind_inv[ind[i]] = i;	
	for (i=0; i<nx; i++) x_tmp[i] = x[ind_inv[i]];
	for (i=0; i<nx; i++) x[i]     = x_tmp[i];
	
	free(x_tmp);
	free(ind_inv);
	
	return(nx);
}




/*********************************************************************/
/* function to check that a dataset x has no identical points :		*/
/* version for 1-d data                                              */
/*********************************************************************/
int check_continuity(double *x, int nx)
{	int *indices, nb_identical=0;
	register int i;
	
	indices = (int*)calloc(nx, sizeof(int));
	for (i=0; i<nx; i++) indices[i]=i;
	
	QuickSort_double(x, indices, 0, nx-1); /* we sort the data */
	
	for (i=1; i<nx; i++)	if (x[i]==x[i-1]) nb_identical ++;
		
	unsort_d_i(x, indices, nx);
	free(indices);
	
	return(nb_identical);
}


/*********************************************************************
 * function to check that a dataset x has no identical points :
 * version for nd dimensional data                         
 * returned value is the nb of identical points
 *
 * 2013-04-02 : first version
 *********************************************************************/
int check_continuity_nd(double *x, int nx, int nd)
{	double *x_copy_1d;
    int *indices, is_identical, nb_identical=0;
	register int i, j1, j2, d;
	
	indices   =    (int*)calloc(nx, sizeof(int));
    x_copy_1d = (double*)calloc(nx, sizeof(double));
	for (i=0; i<nx; i++) 
    {   indices[i]   = i;
        x_copy_1d[i] = x[i];
    }
	
	QuickSort_double(x_copy_1d, indices, 0, nx-1); /* we sort the data */
	
	for (i=1; i<nx; i++)	
    {   if (x_copy_1d[i]==x_copy_1d[i-1]) 
        {   j1 = indices[i];
            j2 = indices[i-1];
            
            // check other dimensions :
            d=1; 
            is_identical = 1;
            while ( (d<nd) && (is_identical==1) )
            {   if (x[d*nx + j1] != x[d*nx + j2])
                    is_identical = 0;
                d++;
            }
            nb_identical += is_identical;
        }
    }
        
//	unsort_d_i(x, indices, nx);
    free(x_copy_1d);
	free(indices);
	
	return(nb_identical);
}



/********************************************************************/
/* maximum norm in 2d												*/
/********************************************************************/
double mn(double x1, double y1, double x2, double y2)
{	double dx = fabs(x2-x1);
	double dy = fabs(y2-y1);

	return((dx > dy) ? dx : dy);
}




// 
// float norm_3d_0(float x1, float y1, float z1, float x2, float y2, float z2)
// {	float v[3];
// 	
// 	v[0] = (x1-x2);
// 	v[1] = (y1-y2);
// 	v[2] = (z1-z2);
// 	
// 	return(my_max_float(v,3));
// }
// 
// float norm_3d_1(float x1, float y1, float z1, float x2, float y2, float z2)
// {	
// 	return(fabs(x1-x2)+fabs(y1-y2)+fabs(z1-z2));
// }
// 
// float norm_3d_2(float x1, float y1, float z1, float x2, float y2, float z2)
// {	
// 	return((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) + (z1-z2)*(z1-z2) );
// }
// 
// float norm_2d_0(float x1, float y1, float x2, float y2)
// {	float v[2];
// 	
// 	v[0] = (x1-x2);
// 	v[1] = (y1-y2);
// 	
// 	return(my_max_float(v,2));
// }
// 
// float norm_2d_1(float x1, float y1, float x2, float y2)
// {	
// 	return(fabs(x1-x2)+fabs(y1-y2));
// }
// 
// float norm_2d_2(float x1, float y1, float x2, float y2)
// {	
// 	return( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
// }





double find_min_double(double *x, int nb_pts)
{	register int i;
	double min = x[0];
	
	for (i=1; i<nb_pts; i++) if (x[i]<=min) min = x[i];
	return(min);
}

double find_max_double(double *x, int nb_pts)
{	register int i;
	double max = x[0];
	
	for (i=1; i<nb_pts; i++) if (x[i]>=max) max = x[i];
	return(max);
}

int find_min_int(int *x, int nx)		/* find the min value in an integer pointer */
{	int mm=x[0];
	register int i;
	
	for (i=1; i<nx; i++) if (x[i]<mm) mm=x[i];
	return(mm);
}

int find_max_int(int *x, int nx)		/* find the max value in an integer pointer */
{	int mm=x[0];
	register int i;
	
	for (i=1; i<nx; i++) if (x[i]>mm) mm=x[i];
	return(mm);
}



double find_mean(double *x, int nb_pts)
{	register int i;
	double m=0.0;
	
	for (i=0; i<nb_pts; i++) m+=x[i];
   	return(m/(double)nb_pts);
}

double find_sigma(double *x, int nb_pts, double mean)
{	register int i;
	double s=0;

	for (i=0; i<nb_pts; i++) s+=(x[i]-mean)*(x[i]-mean);
   	s/=(double)nb_pts;
	return(sqrt(s));
}



/* computes a power of 2, returns 2^p */
int pow2(int p)
{	if (p>0) return(2*pow2(p-1));
	else     return(1);
}





/* average data along time axis, and reduce number of points in time : */
/* average is performed over na consecutive points					   */
/* returns the averaged data										   */
/* returns the number of points after average (<= nx/na) in variable n_out */
float *average_data(float *x, int nx, int na, int *n_out)
{	register int i, j;
	float *y;
	int   n;
	
	n = (int)(nx/na);
	*n_out = n;
	
	y = (float*)calloc(n, sizeof(float));
	
	for (i=0; i<n; i++)
	{	for (j=0; j<na; j++)
	{	y[i] += x[i*na+j];
		}
		y[i] /= na;
	}   
		 
	return(y);
}


/************************************************************************/
/* average data along time axis, and reduce number of points in time    */
/*																		*/
/* x      : data to time-average, multidimensional of size (m,N_pts)	*/
/* N_pts  : nb of points in time										*/
/* m   	  : dimensionality of data										*/
/* tau    : nb of consecutive points to average							*/
/* fr     : how many pts per interval of size tau to keep (resampling)	*/
/*          fr = 1   : keep 1 pt every tau points						*/
/*		    fr = tau : keep all points from the original data			*/
/* 																		*/
/* result is returned in the parameter "out" (must be pre-allocated!)	*/
/* npts_new is the nb of pts in time in the out pointer (pre-allocated) */
/*																		*/
/* returns the number of points after average (<= nx/na) 				*/
/*																		*/
/* 2023-01-25 - new function, see brouillon								*/
/* 2023-02-08 - added parameter npts_new for output size				*/
/************************************************************************/
int filter_FIR_LP(double *x, int N_pts, int m, int tau, double fr, double *out, int npts_new)
{	register int i, j, k, ind;
	int npts_new_adjusted=(int)floor((N_pts-tau+1)*fr/tau);
	int	na = (int)floor(tau/fr); // time shift between 2 new points
	
	if (na<=0) return(0); 
//	printf("npts_new : %d (imposed) vs %d (estimated) - na = %d\n", npts_new, npts_new_adjusted, na);
	
	for (k=0; k<m; k++)	// loop on dimensions
	{	for (i=0; i<npts_new_adjusted; i++)
		{	out[i + k*npts_new] = 0.;
			ind = i*na;
			for (j=0; j<tau; j++)
			{	out[i + k*npts_new] += x[ ind + j + k*N_pts];
			}
			out[i + k*npts_new] /= tau;
		}
	}   
	
	return(npts_new_adjusted);
}



/* determinant of a square matrix M of size n x n:		*/
// my cnvention is: M[i+n*j] = M[i][j]
// to do: (check if C-major or not)
double determinant(double *M, int n)
{	register int i,j,k;
	double det=1.0, ratio;
	
	// using Gauss elimination technique for transforming M into an upper triangular matrix:
	for (i=0; i<n; i++)
	{  	if (M[i+i*n]==0.0) return(0);
		for (j=i+1; j<n; j++)
		{	ratio = M[j+i*n] / M[i+i*n];
			for (k=0; k<n; k++)
				M[j+k*n] = M[j+k*n] - ratio*M[i+k*n];
		}
	}

	// determinant is then the product of diagonal elements:
	for (i=0; i<n; i++)	det *= M[i+i*n];
    
    return(det);
}
