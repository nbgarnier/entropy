/*
Kolmogorov Complexities for Matlab
2014-04-02 forked from "compute_ApEN", uses new function (returns both ApEn and SampEn)
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include "matrix.h"

#include "entropy_others.h"

#define malloc mxMalloc
#define calloc mxCalloc
#define free   mxFree
#define printf mexPrintf

extern int    compute_complexity(double* data, int npts, int m, int stride, double r, int kernel_type, double *ApEn, double *SampEn);

void mexFunction(int nlhs, mxArray *plhs[], int nrhs,const mxArray  *prhs[])
{   double *v_m;
    double *ApEn, *SampEn;
    int nx, n, m=2, stride=1, kernel_type=KERNEL_BRICKWALL;
    int size1, size2;
    int i, j;
    double *x, r=0.25, std=1.0;

    /* verify number of inputs */
    if (nrhs < 1)
    mexErrMsgTxt("There is at least 1 input arguments required. Please specify at least one vector x. "
				"See help for more information.\n");
    if (nrhs > 6) 
    mexErrMsgTxt("Wrong number of input parameters, this function requires 1, 2, 3, 4, 5 or 6 parameters. "
				"See help for more information.\n");

    /* input the data (1st parameter) */
    if(mxIsEmpty(prhs[0])) mexErrMsgTxt("Input signal is empty!!\n");
    if (!(mxIsDouble(prhs[0]))) mexErrMsgTxt("data must be of type double."); /* refuse complex data*/

    v_m=mxGetPr(prhs[0]);
    size1 = mxGetN(prhs[0]); 
    size2 = mxGetM(prhs[0]);

    /* in case of nD we have to reorder the data */
    x=mxMalloc(size1*size2*sizeof(double));
    if (size1 < size2)
    {   n=size1; nx=size2;
        x=v_m;
    } 
    else 
    {   n=size2; nx=size1;
        m=0;
        for (i=0;i<n;i++)
        for (j=0;j<nx;j++)
        {   x[m]=v_m[(j*n)+i];
            m++;
        }
    }
    if (n!=1) mexErrMsgTxt("OK... you want to work on multidimensional data. This can be done, for sure,"
                " but it is not fully implemented here (by lack of interest...) contact NG.\n");
 
    /* second input parameter should be the largest embedding dimension */
    if (nrhs > 1) 
    {   if (mxIsEmpty(prhs[1])) mexErrMsgTxt("Input m is empty!!\n");
        m = (int)mxGetScalar(prhs[1]);
        if (m < 0) mexErrMsgTxt("m must be positive.\n");
    } 
    else m=2;
 
    /* 3rd input parameter is the stride */
    if (nrhs > 2) 
    {   if (mxIsEmpty(prhs[1])) mexErrMsgTxt("Input stride is empty!!\n");
        stride = (int)mxGetScalar(prhs[1]);
        if (stride <= 0) mexErrMsgTxt("stride must be positive.\n");
    } 
    else m=2;
 
    /* 4th input parameter is the radius of the ball / the width of the kernel */
    if (nrhs > 3)
    {   if (mxIsEmpty(prhs[3])) mexErrMsgTxt("Input r is empty!!\n");
        r  = mxGetScalar(prhs[3]);
        if (r <= 0) mexErrMsgTxt("radius r must be positive.\n");
	}
    
    /* 5th input parameter is the standard deviation of the data */
    if (nrhs > 4)
    {   if (mxIsEmpty(prhs[4])) mexErrMsgTxt("Input std is empty!!\n");
        std = mxGetScalar(prhs[4]);
        if (std <= 0) mexErrMsgTxt("std must be positive.\n");
    }
  
    /* 6th input parameter is the kernel type */
    if (nrhs > 5)
    {   if (mxIsEmpty(prhs[5])) mexErrMsgTxt("Input 'kernel type' is empty!!\n");
        kernel_type = (int)mxGetScalar(prhs[5]);
        if ( (kernel_type != 0) && (kernel_type != 1) ) 
            mexErrMsgTxt("kernel type must be either 0 or 1 (as of 2014/04/02).\n");
    }
  

// output arguments / returned values :   
    plhs[0] = mxCreateDoubleMatrix(1, m+1, mxREAL);
    ApEn    = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix(1, m+1, mxREAL);
    SampEn  = mxGetPr(plhs[1]);
    
    if (compute_complexity(x, nx, m, stride, r, kernel_type, ApEn, SampEn)==0)
        nlhs = 2;
}
